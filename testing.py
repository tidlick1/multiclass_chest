# USAGE
# python train.py --dataset D:\deep_learning\cxr_dl\data\chest8\images --dataframe D:\deep_learning\cxr_dl\data\chest8\Data_Entry_2017_v2020.csv --model chest100720.model --labelbin mlb.pickle
# This version has been updated to perform chest x-ray multilevel classification.Original code from
# pyimage search. Downloaded 10/7/2021

# set the matplotlib backend so figures can be saved in the background
import matplotlib
from tensorflow_core.python.keras.callbacks import TensorBoard

matplotlib.use("Agg")
# import the necessary packages
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from pyimagesearch.smallervggnet import SmallerVGGNet
from pyimagesearch.check_for_leakage import check_for_leakage
from pyimagesearch.get_train_generator import get_train_generator
from pyimagesearch.get_test_and_valid_generator import get_test_and_valid_generator
from pyimagesearch.compute_class_freqs import compute_class_freqs
from pyimagesearch.weighted_loss import get_weighted_loss
#from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import pandas as pd
import datetime


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-f", "--dataframe", required=True,
                help="path to input dataframe (i.e., dataframe linking images)")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
                help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 5
INIT_LR = 1e-3
BS = 128
IMAGE_DIMS = (128, 128, 1)

# disable eager execution
#tf.compat.v1.disable_eager_execution()

# Preprocess the data frame
all_data = pd.read_csv(args["dataframe"])
# create list of multi-labels; seperates out into multiple labels
findings_list = all_data['Finding Labels'].apply(lambda x: list(x.split("|")))
# Converting it into dataframe and working on it separately
findings_df = pd.DataFrame({"Findings": findings_list})
# instantiating MultiLabelBinarizer
mlb = MultiLabelBinarizer()
findings_encoded = pd.DataFrame(mlb.fit_transform(findings_df["Findings"]), columns=mlb.classes_)
# Concatenating df and types_encoded to get the final data frame
encoded_cxr_data = pd.concat([all_data, findings_encoded], axis=1)
# clean out the input table by droping some columns
encoded_cxr_data = encoded_cxr_data.drop(['Finding Labels', 'Follow-up #',
                                          'OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x', 'y]'],
                                         axis=1)

# Splits the groups according to patient ID to avoid leakage.
train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state=7).split(encoded_cxr_data,
                                                                                                groups=encoded_cxr_data[
                                                                                                    'Patient ID']))
train_df = encoded_cxr_data.iloc[train_inds]
test_df = encoded_cxr_data.iloc[test_inds]

train_inds, val_inds = next(
    GroupShuffleSplit(test_size=.20, n_splits=2, random_state=7).split(train_df, groups=train_df['Patient ID']))
val_df = train_df.iloc[val_inds]
train_df = train_df.iloc[train_inds]
train_labels = train_df.drop(['Image Index', 'Patient ID', 'Patient Age', 'Patient Gender',
                              'View Position'], axis=1)

# check for leakage in the training and validation sets.
print("[INFO] checking for leakage between :")
print("leakage between train and test: {}".format(check_for_leakage(train_df, test_df, 'Patient ID')))
print("leakage between train and val: {}".format(check_for_leakage(train_df, val_df, 'Patient ID')))
print("leakage between val and test: {}".format(check_for_leakage(val_df, test_df, 'Patient ID')))

# set up the data generator
train_generator = get_train_generator(train_df, (args["dataset"]), "Image Index", mlb.classes_, batch_size=BS, target_w=IMAGE_DIMS[0], target_h=IMAGE_DIMS[1])
valid_generator, test_generator = get_test_and_valid_generator(val_df, test_df, train_df, (args["dataset"]),
                                                               "Image Index", mlb.classes_, batch_size=BS, target_w=IMAGE_DIMS[0], target_h=IMAGE_DIMS[1])

model = tf.keras.models.load_model(args["model"])
# loss, acc = model.evaluate(test_generator, verbose=2)
# print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
test_X, test_Y =next(test_generator)
pred_Y = model.predict(test_X, verbose = True)

from sklearn.metrics import roc_curve, auc
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(mlb.classes_):
    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('trained_net.png')
