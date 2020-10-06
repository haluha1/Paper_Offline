from __future__ import division
import pprint
import os, sys
import random, time
import pickle
import numpy as np
import pandas as pd
from optparse import OptionParser

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras.utils import generic_utils
from Faster_RCNN import config, data_generators, losses, utility
from Faster_RCNN import parse_data
from Faster_RCNN import vgg16 as nn

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-a", "--annotation", dest="annotation", help="Path to training data (annotation.txt).")
parser.add_option("-d", "--dataframe", dest="dataframe", help="Path to training data (csv file).")
parser.add_option("-j", "--json", dest="json", help="Path to training data folder.")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or csv or json (load from json). (Default='simple')", default="simple")
parser.add_option("-i", "--img_folder", dest="img_folder", help="Location of folder where contain train image.(Default='dataset'", default="dataset")
parser.add_option("-s", "--save_data_folder", dest="save_data_folder", help="Location to store parse_data.(Default='data')", default="data")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.(Default=4)", default=4)
parser.add_option("--network", dest="network", help="Base network to use. Only support vgg16 now.(Default='vgg16')", default='vgg16')
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).", action="store_true", default=False)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=100)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to store all the metadata related to the training (to be used when testing).",
				default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='model/model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")
parser.add_option("--svm", dest="svm", help="Continue training svm or not. (Default=false).", action="store_true", default=False)

(options, args) = parser.parse_args()

if options.parser == 'simple':
    if not options.annotation:
        parser.error('Error: path to training data must be specified. Pass --annotation to command line')
elif options.parser == 'csv':
    if not options.dataframe:
        parser.error('Error: path to training data must be specified. Pass --dataframe to command line')
elif options.parser == 'json':
    if not options.json:
        parser.error('Error: path to training data folder must be specified. Pass --json to command line')
else:
    raise ValueError("Command line option parser must be one of 'simple' or 'csv' or 'json' for loading parsed_data")

# pass the settings from the command line, and persist them in the config object
C = config.Config()

C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

C.model_path = options.output_weight_path
if C.model_path[-5:] != '.hdf5':
    print('Output weights must have .hdf5 filetype')
    exit(1)
C.num_rois = int(options.num_rois)
C.network = 'vgg'
record_path = 'model/record.csv'
C.record_path = record_path


# check if weight path was passed via command line
if options.input_weight_path:
    C.base_net_weights = options.input_weight_path
else:
    # set the path to weights based on backend and model
    C.base_net_weights = nn.get_weight_path()

train_imgs, classes_count, class_mapping = None, None, None
if options.parser == 'csv':
    df_train = pd.read_csv(options.dataframe)
    save_anno_file = os.path.join(options.save_data_folder,"annotation.txt")
    parse_data.df_to_annotation(df_train, save_path=save_anno_file, img_folder=options.img_folder)
    train_imgs, classes_count, class_mapping = parse_data.get_data(save_anno_file)
if options.parser == 'simple':
    train_imgs, classes_count, class_mapping = parse_data.get_data(options.annotation)
if options.parser == 'json':
    train_imgs, classes_count, class_mapping = parse_data.load_data_json(save_folder=options.json)

parse_data.save_data_json(train_imgs, classes_count, class_mapping, save_folder=options.save_data_folder)
if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping
class_mapping_inv = {v: k for k, v in C.class_mapping.items()}
C.class_mapping_inv = class_mapping_inv

print('Training images per class:')
pprint.pprint(classes_count)
print(f'Num classes (including bg) = {len(classes_count)}')

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C,config_f)
    print(f'Config has been written to {config_output_filename}, and can be loaded when testing to ensure correct results')

random.seed(1)
random.shuffle(train_imgs)
num_imgs = len(train_imgs)

#train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
#val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print(f'Num train samples {len(train_imgs)}')


data_gen_train = data_generators.get_anchor_gt(train_imgs, C, nn.get_img_output_length, mode='train')

input_shape_img = (None, None, 3)
if K.common.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios) # 9
rpn = nn.rpn_layer(shared_layers, num_anchors)
classifier = nn.classifier_layer(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count))

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

if not os.path.isfile(record_path):
    try:
        print(f'Loading weights from {C.base_net_weights}')
        model_rpn.load_weights(C.base_net_weights, by_name=True)
        model_classifier.load_weights(C.base_net_weights, by_name=True)
    except:
        print('Could not load pretrained model weights. Weights can be found in the keras application folder \
                https://github.com/fchollet/keras/tree/master/keras/applications')
    record_df = pd.DataFrame(columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])
else:
    # If this is a continued training, load the trained model from before
    print('Continue training based on previous trained model')
    print('Loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)
    
    # Load the records
    record_df = pd.read_csv(record_path)

    r_mean_overlapping_bboxes = record_df['mean_overlapping_bboxes']
    r_class_acc = record_df['class_acc']
    r_loss_rpn_cls = record_df['loss_rpn_cls']
    r_loss_rpn_regr = record_df['loss_rpn_regr']
    r_loss_class_cls = record_df['loss_class_cls']
    r_loss_class_regr = record_df['loss_class_regr']
    r_curr_loss = record_df['curr_loss']
    r_elapsed_time = record_df['elapsed_time']
    r_mAP = record_df['mAP']
    print('Already train %dK batches'% (len(record_df)))
    
optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

total_epochs = len(record_df)
r_epochs = len(record_df)

epoch_length = 1000
num_epochs = int(options.num_epochs)
iter_num = 0

total_epochs += num_epochs

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []

if len(record_df)==0:
    best_loss = np.Inf
else:
    best_loss = np.min(r_curr_loss)

print('Starting training')

##os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
start_time = time.time()
for epoch_num in range(num_epochs):

    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(r_epochs + 1, total_epochs))
    
    r_epochs += 1

    while True:
        try:
            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                #print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            # Generate X (x_img) and label Y ([y_rpn_cls, y_rpn_regr])
            X, Y, img_data, debug_img, debug_num_pos = next(data_gen_train)

            # Train rpn model and get loss value [_, loss_rpn_cls, loss_rpn_regr]
            loss_rpn = model_rpn.train_on_batch(X, Y)

            # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
            P_rpn = model_rpn.predict_on_batch(X)

            # R: bboxes (shape=(300,4))
            # Convert rpn layer to roi bboxes
            R = utility.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.common.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
            
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            # X2: bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
            # Y1: one hot code for bboxes from above => x_roi (X)
            # Y2: corresponding labels and corresponding gt bboxes
            X2, Y1, Y2, IouS = utility.calc_iou(R, img_data, C, class_mapping)

            # If X2 is None means there are no matching bboxes
            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue
            
            # Find out the positive anchors and negative anchors
            # (IoU) > 0.5 are considered “foreground” and those that don’t overlap any ground truth object
            # or have less than 0.1 IoU with ground-truth objects are considered “background”.
            # foregrounds (pos) and backgrounds (neg)
            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if C.num_rois > 1:
                # If number of positive anchors is larger than 4//2 = 2, randomly choose 2 pos samples
                if len(pos_samples) < C.num_rois//2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                
                # Randomly choose (num_rois - num_pos) neg samples
                try:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
                
                # Save all the pos and neg samples in sel_samples
                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            # training_data: [X, X2[:, sel_samples, :]]
            # labels: [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
            #  X                     => img_data resized image
            #  X2[:, sel_samples, :] => num_rois (4 in here) bboxes which contains selected neg and pos
            #  Y1[:, sel_samples, :] => one hot encode for num_rois bboxes which contains selected neg and pos
            #  Y2[:, sel_samples, :] => labels and gt bboxes for num_rois bboxes which contains selected neg and pos
            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            iter_num += 1

            progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                      ('final_cls', np.mean(losses[:iter_num, 2])), ('final_regr', np.mean(losses[:iter_num, 3]))])

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Total loss: {}'.format(loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))
                    elapsed_time = (time.time()-start_time)/60

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                    best_loss = curr_loss
                    model_all.save_weights(C.model_path)

                new_row = {'mean_overlapping_bboxes':round(mean_overlapping_bboxes, 3), 
                           'class_acc':round(class_acc, 3), 
                           'loss_rpn_cls':round(loss_rpn_cls, 3), 
                           'loss_rpn_regr':round(loss_rpn_regr, 3), 
                           'loss_class_cls':round(loss_class_cls, 3), 
                           'loss_class_regr':round(loss_class_regr, 3), 
                           'curr_loss':round(curr_loss, 3), 
                           'elapsed_time':round(elapsed_time, 3), 
                           'mAP': 0}

                record_df = record_df.append(new_row, ignore_index=True)
                record_df.to_csv(record_path, index=0)

                break

        except Exception as e:
            print('Exception: {}'.format(e))
            continue

if bool(options.svm):
    print('Training faster R-CNN complete, training svm.')
    out_fea = model_classifier.get_layer(model_classifier.layers[-4].name).output
    model2 = Model(model_classifier.input, out_fea)

    X_train = []
    y_train = []
    start_time = time.time()
    for epoch_num in range(len(train_imgs)):
        sys.stdout.write('\r'+'Preparing data: {}/{}'.format(epoch_num+1,len(train_imgs)))
        r_epochs += 1
        try:
            X, Y, img_data, debug_img, debug_num_pos = next(data_gen_train)
            P_rpn = model_rpn.predict_on_batch(X)
            R = utility.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.common.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
            X2, Y1, Y2, IouS = utility.calc_iou(R, img_data, C, class_mapping)
            if X2 is None:
                continue
            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            if C.num_rois > 1:
                if len(pos_samples) < C.num_rois//2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                
                try:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
                
                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            P_feature = model2.predict([X, X2[:, sel_samples, :]])
            X_train.append(P_feature)
            y_train.append(Y1[:, sel_samples, :])
            iter_num += 1
        except Exception as e:
            print('Exception: {}'.format(e))
            continue

    tmp_X = np.asarray(X_train)
    new_shape = 1
    for tmp_shape in tmp_X.shape[:-1]:
        new_shape = new_shape * tmp_shape
    tmp_X = tmp_X.reshape((new_shape, 4096))
    tmp_y = np.asarray(y_train)
    tmp_y = tmp_y.reshape((tmp_y.shape[0]*tmp_y.shape[1]*tmp_y.shape[2],7))
    tmp_y = [class_mapping_inv[np.argmax(i)] for i in tmp_y]
    del X_train
    del y_train
    
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import cross_val_score, GridSearchCV
    params_grid = [{'loss': ['hinge', 'log', 'perceptron', 'modified_huber'], #perceptron,log
                'alpha': [1e-3, 1e-4, 1e-5],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'random_state': [2020],
                'max_iter': [1000],
                'n_jobs': [-1]}]
    svm_model = GridSearchCV(SGDClassifier(), params_grid, cv=2)
    print("Training svm...")
    svm_model.fit(tmp_X, tmp_y)
    from sklearn.externals import joblib
    save_SGD = "model/SVM_SGD.pkl"
    print(f"Saving svm to {save_SGD}")
    if os.path.exists(save_SGD):
        os.remove(save_SGD)
    joblib.dump(svm_model, save_SGD)
print('Training complete, exiting.')
