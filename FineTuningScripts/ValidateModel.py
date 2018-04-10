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

from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.preprocessing import image
from scipy.misc import imresize
from keras import backend as K
from keras import optimizers
from skimage import io

#########
#
# Check that the restored model has the same validation accuracy as when
# it was saved during training.
#
# During training, the everage accuracy, over batches, is dysplayed.
# So, at test time, there could a small difference.
#
#########

parser = argparse.ArgumentParser()

parser.add_argument("--model", help = "Model to Validate", required = True)
  
arguments = vars(parser.parse_args())

mod = arguments["model"]

#########

GPU = GPUtil.getAvailable(order = "memory")[0]
GPU = str(GPU)

os.environ["CUDA_VISIBLE_DEVICES"] = GPU

#########

MODEL = "./models/" + str(mod)

model = load_model(MODEL)

print("\nValidating: %s" %(MODEL), "\n")

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

print("\n")

valid_images = np.stack(valid_images)
valid_images = (valid_images/255).astype(np.float32) ### Standardise

valid_labels = np.array(valid_labels).astype(np.int32)
Y_valid = to_categorical(valid_labels, num_classes = np.unique(valid_labels).shape[0])

### 

valid_loss, valid_accuracy = model.evaluate(valid_images, Y_valid, batch_size = 50)

print("\nValidation Loss: %.3f" %(valid_loss))
print("Validation Accuracy: %.3f" %(valid_accuracy))
print("=" * 30, "\n")

print("Classification Report")
print("=" * 30, "\n")

posteriors = model.predict(valid_images, batch_size = 32)
predictions = np.argmax(posteriors, axis = 1)

cr = classification_report(valid_labels, predictions, target_names = ["CYT", "FIB", "HOF", "SYN", "VEN"], digits = 3)
print(cr, "\n")

print("Confusion Matrix")
print("=" * 30, "\n")

cm = confusion_matrix(valid_labels, predictions)
print(cm, "\n")

### Save Confusion Matrix

now = datetime.datetime.now()

day, month, year, hour, minute = now.day, now.month, now.year, now.hour, now.minute

date = str(day) + "-" + str(month) + "-" + str(year) + "-" + str(now.hour) + ":" + str(now.minute) + "-" + mod

cm_name = "./confusion/" + date + "-conf-matrix.npy"

# np.save(cm_name, cm)

####

classes = ["CYT", "FIB", "HOF", "SYN", "VEN"]

print("\nAccuracy per Class")
print("=" * 30, "\n")

for index, CLASS in enumerate(classes):
    correct = cm[index, index]
    all = cm[index, :].sum()
    acc = (correct/all) * 100
    print("%s: %.1f%%" %(classes[index], acc))

print("\n")

K.clear_session()

