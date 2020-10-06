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
df_test = pd.DataFrame()
if options.parser == 'simple':
    if not options.annotation:
        parser.error('Error: path to training data must be specified. Pass --annotation to command line')
    else:
        imgs_test, test_count, _ = parse_data.get_data(options.annotation)
        labels_test = [img_data['bboxes']['class'] for idx, img_data in enumerate(imgs_test)]
        df_test['name'] = imgs_test
        df_test['label'] = labels_test
elif options.parser == 'csv':
    if not options.dataframe:
        parser.error('Error: path to training data must be specified. Pass --dataframe to command line')
    else:
        df_test = pd.read_csv(options.dataframe)
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

F_model = model.FasterRCNN(C,num_features=512,svm=final_model)
vehicle_db = database.Database(n_dim=4096, distance="euclidean")
vehicle_db.create_db(df_test, F_model, base_path=options.img_folder)
vehicle_db.save_db()
