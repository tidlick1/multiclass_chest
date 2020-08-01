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
import pickle
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
EPOCHS = 1
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

# loop over each of the possible class labels and show them
print("[INFO] The following classes were identified:")
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i + 1, label))

# set up the data generator
train_generator = get_train_generator(train_df, (args["dataset"]), "Image Index", mlb.classes_, batch_size=BS, target_w=IMAGE_DIMS[0], target_h=IMAGE_DIMS[1])
valid_generator, test_generator = get_test_and_valid_generator(val_df, test_df, train_df, (args["dataset"]),
                                                               "Image Index", mlb.classes_, batch_size=BS, target_w=IMAGE_DIMS[0], target_h=IMAGE_DIMS[1])

#calculate the positve and negative class frequencies for the weighted loss function
freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
pos_weights = freq_neg
neg_weights = freq_pos

# Calculate the class weights given the significant class imbalance.
y_train = train_labels.values  # this is a numpy array of
y_ints = [y.argmax() for y in y_train]

class_weights = class_weight.compute_class_weight('balanced', np.unique(y_ints), y_ints)
class_weights_dict = dict(enumerate(class_weights))

#class_weights_dict = dict(enumerate(neg_weights))

# reduce lr on plateau
reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.7,
    patience=3,
    min_lr=1e-6,
    verbose=1,
    min_delta=1e-3
)

# # save the model to disk
# print("[INFO] serializing network...")
# model.save(args["model"])
#
# # if not os.path.isdir(save_dir):
# #     os.makedirs(save_dir)
# # filepath = os.path.join(save_dir, model_name)

# checkpoint callback
checkpoint = tf.keras.callbacks.ModelCheckpoint(args["model"],
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

# Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=8,
    verbose=1,
    min_delta=1e-3,
    restore_best_weights=True
)

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]

#setup tensorboard
log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

test_shape = train_generator.image_shape

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
    width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
    depth=train_generator.image_shape[2], classes=len(mlb.classes_),
    finalAct="sigmoid")

# initialize the optimizer (SGD is sufficient)
#opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
opt = Adam(lr=INIT_LR)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
# model.compile(loss=get_weighted_loss(pos_weights, neg_weights), optimizer=opt,
#               metrics=METRICS)

model.compile(loss=get_weighted_loss(pos_weights,neg_weights), optimizer='adam',
               metrics=['binary_accuracy', 'mae', 'AUC'])


# train the network
print("[INFO] training network...")
H = model.fit_generator(
    generator=train_generator,
    validation_data=valid_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    validation_steps=valid_generator.n // valid_generator.batch_size,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[early_stopping, checkpoint, tensorboard_callback])
#

# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(mlb))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
