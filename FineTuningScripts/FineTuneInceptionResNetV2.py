import pandas as pd
import numpy as np
import argparse
import datetime
import GPUtil
import random
import keras
import glob
import time
import sys
import os

from keras.layers import Dense, GlobalAveragePooling2D, Dropout, MaxPooling2D, Flatten
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from sklearn.metrics import classification_report, confusion_matrix
from keras.initializers import Orthogonal
from keras.utils import to_categorical
from keras.preprocessing import image
from generators import DataGenerator
from scipy.misc import imresize
from keras.models import Model
from keras import backend as K
from keras import optimizers
from skimage import io

#########
#
# Script with flags:
#
# python script.py ---> Does NOT save logs and best model (as default is False).
#
# python script.py --save=True ---> save logs and best model (based on best validation accuracy).
#
# Automatically detects the GPU with lowest memory usage and deploys the script there. The script
# will be deployed only if there's a GPU with memory usage less than 0.5 (threshold can be
# changed). 
#
#########

parser = argparse.ArgumentParser(description = "Fine Tune ???") 

parser.add_argument("--save", help = "Save Logs and Best Model", default = "False")

arguments = vars(parser.parse_args())

script_name = sys.argv[0].split(".")[0]

GPU = GPUtil.getAvailable(order = "memory")[0]
GPU = str(GPU)

os.environ["CUDA_VISIBLE_DEVICES"] = GPU

#########

base = InceptionResNetV2(weights = "imagenet", include_top = False, input_shape = (299, 299, 3))

#########
#
# On top of the feature extraction module, add a neural network classifier with one 
# hidden layer.
#
# Output of HFE module is 2048D - go down to 2048/4 = 512D.
#
#########

x = base.output

x = MaxPooling2D()(x)

x = Flatten()(x)

x = Dropout(rate = 0.5)(x)

x = Dense(128, activation = "relu", kernel_initializer = Orthogonal())(x)

x = Dropout(rate = 0.5)(x)

predictions = Dense(5, activation = "softmax", kernel_initializer = Orthogonal())(x)

model = Model(inputs = base.input, outputs = predictions)

#########
 
for layer in model.layers:
    layer.trainable = True

#########

model.compile(optimizer = "SGD", loss = "categorical_crossentropy", metrics = ["accuracy"])

#########

trainable_params = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))

non_trainable_params = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

print("\nModel Stats")
print("=" * 30)
print("Total Parameters: {:,}".format((trainable_params + non_trainable_params)))
print("Non-Trainable Parameters: {:,}".format(non_trainable_params))
print("Trainable Parameters: {:,}\n".format(trainable_params))

#########

### Create training data set by oversampling all classes to the size of the majority class ###

### Classes in alphabetical order; CYT = 0, FIB = 1, HOF = 2, SYN = 3, VEN = 4 ###

### TRAINING ###

train_folders = ["./data/train/CYT/", "./data/train/FIB/", "./data/train/HOF/", "./data/train/SYN/", "./data/train/VEN/"]

population_sizes = []

print("\nImages for Training")
print("=" * 30)

for folder in train_folders:
    files = glob.glob(folder + "*.png")
    n = len(files)
    print("Class: %s. " %(folder.split("/")[-2]), "Size: {:,}".format(n))
    population_sizes.append(n)

MAX = max(population_sizes)

train_images = []
train_labels = []

for index, folder in enumerate(train_folders):
     files = glob.glob(folder + "*.png")
     sample = list(np.random.choice(files, MAX))
     images = io.imread_collection(sample)
     images = [imresize(image, (299, 299)) for image in images] ### Reshape to (299, 299, 3) ###
     labels = [index] * len(images)
     train_images = train_images + images
     train_labels = train_labels + labels

train_images = np.stack(train_images)
train_images = (train_images/255).astype(np.float32) ### Standardise into the interval [0, 1] ###

train_labels = np.array(train_labels).astype(np.int32)
Y_train = to_categorical(train_labels, num_classes = np.unique(train_labels).shape[0])

### VALIDATION ###

valid_folders = ["./data/validation/CYT/", "./data/validation/FIB/", "./data/validation/HOF/", "./data/validation/SYN/", "./data/validation/VEN/"]

print("\nImages for Validation")
print("=" * 30)

valid_images = []
valid_labels = []

for index, folder in enumerate(valid_folders):
    files = glob.glob(folder + "*.png")
    images = io.imread_collection(files)
    images = [imresize(image, (299, 299)) for image in images] ### Reshape to (299, 299, 3) ###
    labels = [index] * len(images)
    valid_images = valid_images + images
    valid_labels = valid_labels + labels
    print("Class: %s. Size: %d" %(folder.split("/")[-2], len(images)))

valid_images = np.stack(valid_images)
valid_images = (valid_images/255).astype(np.float32) ### Standardise

valid_labels = np.array(valid_labels).astype(np.int32)
Y_valid = to_categorical(valid_labels, num_classes = np.unique(valid_labels).shape[0])

### 

print("\nBootstrapping to Balance - Training set size: %d (%d X %d)" %(train_labels.shape[0], MAX, np.unique(train_labels).shape[0]))
print("=" * 30, "\n")

### Data generators ###

### TRAINING ###

n_epochs = 20

batch_size_for_generators = 50

train_datagen = DataGenerator(rotation_range = 178, horizontal_flip = True, vertical_flip = True, shear_range = 0.6, stain_transformation = True)

train_gen = train_datagen.flow(train_images, Y_train, batch_size = batch_size_for_generators)

### VALIDATION ###

valid_datagen = DataGenerator()

valid_gen = valid_datagen.flow(valid_images, Y_valid, batch_size = batch_size_for_generators)

#########

start = time.time()

#########

now = datetime.datetime.now()

day, month, year, hour, minute = now.day, now.month, now.year, now.hour, now.minute

date = str(day) + "-" + str(month) + "-" + str(year) + "-" + str(now.hour) + ":" + str(now.minute)

logs_path = "./logs/" + date + "-" + script_name + "-train-history.csv"

csv_logger = CSVLogger(logs_path, append = True, separator = ",")

### Define the checkpoint to save the model with the highest validation accuracy ###

filepath = "./models/" + date + "-" + script_name + "-best-model.h5"

checkpoint = ModelCheckpoint(filepath, monitor = "val_acc", verbose = 1, save_best_only = True, mode = "max")

#########

callbacks_list = [csv_logger, checkpoint]

#########

train_steps = train_images.shape[0]//batch_size_for_generators

valid_steps = valid_images.shape[0]//batch_size_for_generators

if (arguments["save"] == "True"):
    print("\nSaving Training Logs.")
    print("=" * 30, "\n")
    model.fit_generator(generator = train_gen, epochs = n_epochs, steps_per_epoch = train_steps, validation_data = valid_gen, validation_steps = valid_steps, callbacks = callbacks_list)
else:
    model.fit_generator(generator = train_gen, epochs = n_epochs, steps_per_epoch = train_steps, validation_data = valid_gen, validation_steps = valid_steps)

#########

delta = time.time() - start
hours = delta/3600

print("\nTraining Time (%d Epochs): %.3f seconds (%.3f hours)." %(n_epochs, delta, hours))
print("=" * 30, "\n")

K.clear_session()


