import os, sys
import cv2
import time
import pickle
import numpy as np
import pandas as pd
from optparse import OptionParser

from keras import backend as K
from keras.layers import Input
from keras.models import Model
from Faster_RCNN import config, model, utility
from Faster_RCNN import parse_data, data_generators, database


sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-a", "--annotation", dest="annotation", help="Path to training data (test_annotation.txt).")
parser.add_option("-d", "--dataframe", dest="dataframe", help="Path to training data (csv file).")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or csv. (Default='simple')", default="simple")
parser.add_option("-i", "--img_folder", dest="img_folder", help="Location of folder where contain train image.(Default='dataset')", default="dataset")
parser.add_option("-n", "--num_rois", dest="num_rois", help="Number of ROIs per iteration. Higher means more memory use.(Default=4)", default=4)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training). (Default='config.pickle')",
				default="config.pickle")

parser.add_option("-w", "--output_model_number", dest="model_iter", help="Models of Epoch step to use. Type this with leading spaces for the hdf5 files!"),
parser.add_option("--svm", dest="svm", help="Using svm or not. (Default=false).", action="store_true", default=False)

(options, args) = parser.parse_args()
imgs_test = []
labels_test = []
test_count = 0
if options.parser == 'simple':
    if not options.annotation:
        parser.error('Error: path to training data must be specified. Pass --annotation to command line')
    else:
        imgs_test, test_count, _ = parse_data.get_data(options.annotation)
        labels_test = [img_data['bboxes']['class'] for idx, img_data in enumerate(imgs_test)]
elif options.parser == 'csv':
    if not options.dataframe:
        parser.error('Error: path to training data must be specified. Pass --dataframe to command line')
    else:
        df_test = pd.read_csv(options.dataframe)
        imgs_test = df_test['name']
        labels_test = df_test['label']
        test_count = {k:v for k,v in df_test['label'].value_counts().reset_index().values}
else:
    raise ValueError("Command line option parser must be one of 'simple' or 'csv' or 'json' for loading parsed_data")

config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

if options.model_iter is not None:
    C.model_path = options.model_iter

C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.img_folder

format_img_size = utility.format_img_size
format_img_channels = utility.format_img_channels
format_img = utility.format_img
get_real_coordinates = utility.get_real_coordinates

record_df = pd.read_csv(C.record_path)
r_epochs = len(record_df)

final_model = None
if (bool)(options.svm):
    from sklearn.externals import joblib
    print("Loading svm from model/SVM/SGD.pkl")
    final_model = joblib.load('model/SVM_SGD.pkl')

model = model.FasterRCNN(C,num_features=512,svm=final_model)
vehicle_db = database.Database(n_dim=4096, distance="euclidean")
vehicle_db.load_db()
featureDB = vehicle_db.df

def eval_feat(feat, disitance_function, label=None, threshold=0.5, top_k=5000):
    '''
    Argument:
    img_pred_path:     path to search image
    label(str):   label of img_pred
    threshold:    threshold for euclidean between 2 img vector
                  < threshold : relevant img
                  <= threshold : irrelevant img
    return: (float)
    AP of this query image.
    '''
    AP = 0.0
    Relevant_Img = 0
    PRelevant_Img = 0
    if len(feat) < 1:
        if label is None:
            return 1.0, PRelevant_Img, 0
        else:
            return 0.0, PRelevant_Img, 0
    feature = feat
    vehicle_query_result = vehicle_db.db.get_nns_by_vector(feature, n=top_k)
    selected_result = []
    distance_arr = []
    for i in vehicle_query_result:
        tmp_distance = disitance_function(feature,vehicle_db.db.get_item_vector(i))
        if tmp_distance < threshold:
            selected_result.append(i)
            distance_arr.append(tmp_distance)
    index_arr = np.argsort(distance_arr)
    if len(selected_result) == 0:
        return 0.0, PRelevant_Img, 0

    for i in range(len(selected_result)):
        img_index = selected_result[index_arr[i]]
        img_label = featureDB.loc[img_index]['label']
        if img_label == label:
            Relevant_Img += 1
            AP += (Relevant_Img / (i+1))
            PRelevant_Img += 1

    if Relevant_Img == 0:
        return 0, PRelevant_Img, len(selected_result)
    return AP/Relevant_Img, PRelevant_Img, len(selected_result)

def evaluate(img_paths, image_labels, distance_function, base_path=None, use_svm=False, threshold=0.5, top_k=5000):
    APs = []
    i = 0
    n_sample = len(img_paths)
    relevant_Imgs = {k:0 for k,v in C.class_mapping.items()}
    predicted_Imgs = {k:0 for k,v in C.class_mapping.items()}
    for img_path, img_label in zip(img_paths,image_labels):
        sys.stdout.write('\r'+'idx=' + str(i) + '/' + str(n_sample))
        img_feat = []
        X = model.extract_feature(img_path,base_path=base_path,use_svm=use_svm)
        if len(X) > 0:
            tmp_results = X[np.argmax([X[i][0] for i in range(len(X))])]
            img_feat = tmp_results[1]
        AP, n_relevant, n_predicted = eval_feat(img_feat, distance_function, label=img_label, threshold=threshold, top_k=top_k)
        relevant_Imgs[img_label] += n_relevant
        predicted_Imgs[img_label] += n_predicted
        APs.append(AP)
        i += 1
    return round(np.mean(np.array(APs)), 3), relevant_Imgs, predicted_Imgs

mean_average_prec, rel, pred = evaluate(imgs_test, labels_test, utility.euclidean, base_path=options.img_folder, use_svm=options.svm, threshold=0.7, top_k=5000)
db_count = {k:v for k,v in vehicle_db.df['label'].value_counts().reset_index().values}
num_imgs = sum([v*test_count[k] for k,v in db_count.items() if k!='bg'])
precision = {k:round(100*rel[k]/pred[k],4) for k,v in C.class_mapping.items() if k!='bg'}
recall = {k:round(100*rel[k]/(db_count[k]*test_count[k]),4) for k,v in C.class_mapping.items() if k!='bg'}

precision_all = round(sum([v for k,v in rel.items() if k!='bg'])/sum([v for k,v in pred.items() if k!='bg']), 4)
recall_all = round(sum([v for k,v in rel.items() if k!='bg'])/num_imgs, 4)
print(f"MAP: {mean_average_prec}")
print(f"Precision: {precision_all}")
print(precision)
print(f"Recall: {recall_all}")
print(recall)
