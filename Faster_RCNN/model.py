import os, sys
import cv2
import time
import numpy as np
import pandas as pd

from keras import backend as K
from keras.layers import Input
from keras.models import Model
from Faster_RCNN import utility
from Faster_RCNN import vgg16 as nn


format_img_size = utility.format_img_size
format_img_channels = utility.format_img_channels
format_img = utility.format_img
get_real_coordinates = utility.get_real_coordinates

class FasterRCNN():
    def __init__(self, C, num_features=512, svm=None):
        self.C = C
        self.svm = svm
        self.num_features = num_features
        self.input_shape_img = (None, None, 3)
        self.input_shape_features = (None, None, num_features)

        self.img_input = Input(shape=self.input_shape_img)
        self.roi_input = Input(shape=(self.C.num_rois, 4))
        self.feature_map_input = Input(shape=self.input_shape_features)

        # define the base network (VGG here, can be Resnet50, Inception, etc)
        self.shared_layers = nn.nn_base(self.img_input, trainable=True)

        # define the RPN, built on the base layers
        self.num_anchors = len(self.C.anchor_box_scales) * len(self.C.anchor_box_ratios)
        self.rpn_layers = nn.rpn_layer(self.shared_layers, self.num_anchors)

        self.classifier = nn.classifier_layer(self.feature_map_input, self.roi_input, self.C.num_rois, nb_classes=len(self.C.class_mapping))
        self.model_rpn = Model(self.img_input, self.rpn_layers)
        self.model_classifier_only = Model([self.feature_map_input, self.roi_input], classifier)
        self.model_classifier = Model([self.feature_map_input, self.roi_input], self.classifier)

        print('Loading weights from {}'.format(self.C.model_path))
        self.model_rpn.load_weights(self.C.model_path, by_name=True)
        self.model_classifier.load_weights(self.C.model_path, by_name=True)

        self.model_rpn.compile(optimizer='sgd', loss='mse')
        self.model_classifier.compile(optimizer='sgd', loss='mse')

        out_cls = self.model_classifier_only.get_layer(self.model_classifier_only.layers[-2].name).output
        out_fea = self.model_classifier_only.get_layer(self.model_classifier_only.layers[-4].name).output
        self.model2 = Model(self.model_classifier_only.input, [out_cls,out_fea])

    def extract_feature(self, img_name, base_path=None, use_svm=False, bbox_threshold=0.7, verbose=False):
        ''' The function predict the label and apply the bbox of the vehicles.

        Args:
        img_name: Image path.
        base_path: Folder contain imgs. (Default=None)
        bbox_threshold: If the box classification value is less than this,
        we ignore this box. (Default = 0.7)

        '''
        # Check is image file.
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            return "Wrong format"

        # st = time.time() # Start count time to predict
        filepath = img_name
        bg_cls = len(self.C.class_mapping) - 1
        if use_svm:
            bg_cls = np.where(self.svm.classes_ == 'bg')[0][0]
        if base_path is not None:
            filepath = os.path.join(base_path, img_name)
        # Read the img
        img = cv2.imread(filepath) # img.shape=(1200, 1600, 3)

        st = time.time()
        # Resize img to input model size.
        # Return img resized and the ratio resized.
        # e.g: img(1200, 1600) => img_resized(600,800), ratio = 0.5
        X, ratio = format_img(img, self.C) # X.shape=(1, 3, 600, 800), ratio=0.5
        # Format the img
        X = np.transpose(X, (0, 2, 3, 1)) # X.shape=(1, 600, 800, 3)

        # get output layer Y1, Y2 from the RPN and the feature maps F
        # Y1: y_rpn_cls
        # Y2: y_rpn_regr
        # Y1.shape = (1, 37, 50, 9)
        # Y2.shape = (1, 37, 50, 36)
        # F.shape = (1, 37, 50, 512)
        [Y1, Y2, F] = self.model_rpn.predict(X)

        # Get bboxes by applying NMS 
        # R.shape = (300, 4)
        # 4 = (x1,y1,x2,y2)
        # It mean R contain 300 couple of points (x1,y1,x2,y2)
        R = utility.rpn_to_roi(Y1, Y2, self.C, K.common.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        # bboxes of objects in the img.
        bboxes = {} # e.g: {'Sedan': [0.98768216, 0.7442094, ..., 0.9753626, 0.9117886, 0.9728151]}
        # probability of the object class in bboxes
        probs = {} # e.g: {'Sedan': [[320, 144, 608, 432], [336, 144, 640, 432], ..., [336, 144, 640, 448], [320, 144, 608, 416], [336, 144, 640, 432]]}
        # len(bboxes) = len(probs)
        features = {}

        # Predict bboxes and classname
        for jk in range(R.shape[0]//self.C.num_rois + 1):
            ROIs = np.expand_dims(R[self.C.num_rois*jk:self.C.num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break
            if jk == R.shape[0]//self.C.num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],self.C.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_features] = self.model2.predict([F, ROIs])
            new_shape = 1
            for tmp_shape in P_features.shape[:-1]:
                new_shape *= tmp_shape
            P_features = P_features.reshape((new_shape,4096))
            if use_svm:
                P_cls = self.svm.predict_proba(P_features)

            new_cls_shape = 1
            for tmp_shape in P_cls.shape[:-1]:
                new_cls_shape *= tmp_shape
            P_cls = P_cls.reshape((new_cls_shape,len(self.C.class_mapping)))

            # Calculate bboxes coordinates on resized image
            for ii in range(P_cls.shape[0]):
                # Ignore 'bg' class
                if np.max(P_cls[ii, :]) < bbox_threshold or np.argmax(P_cls[ii, :]) == bg_cls:
                    continue

                cls_name = self.C.class_mapping[np.argmax(P_cls[ii, :])]
                if use_svm:
                    cls_name = self.svm.classes_[np.argmax(P_cls[ii, :])]
                # Add class if not exist in bboxes
                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
                    features[cls_name] = []

                probs[cls_name].append(np.max(P_cls[ii, :]))
                features[cls_name].append(P_features[ii])

        # all_dets contain the results.
        # [('Sedan', 99.34011101722717)]
        all_dets = []

        for key in probs:
            prob = np.array(probs[key])
            for bb in range(prob.shape[0]):
                new_probs = prob[bb]
                feat = features[key][bb]
                all_dets.append((100*new_probs,feat))
          
        if verbose == True:
            print('Elapsed time = {}'.format(time.time() - st))
        return all_dets


