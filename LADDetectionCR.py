from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
import pandas as pd
import glob
import locale
import cv2
import numpy as np
import argparse
import os
import csv
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import array_to_img
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def load_LAD_attributes():
	
	data = open("data.csv")
	csvreader = csv.reader(data)
	
	header = []
	header = next(csvreader)
	
	LADrows = []
	LADcodes = []
	for row in csvreader:
		LADcode = row[0]
		LADcodes.append(LADcode)
		LADattr = row[5:9:]
		LADrows.append(LADattr)
	
	return (LADcodes, LADrows)

def load_LAD_images(LADattr):
	rawlst = []
	
	for code in range(len(LADattr)):
		dataraw = dicom.read_file(RCAattr[code])
		dcm = dataraw.pixel_array.astype(float)
		scaled = (np.maximum(dcm, 0) / dcm.max()) * 255
		datarawarr = np.uint8(scaled)
		rawlst.append(datarawarr)
	
	return rawlst

(LADcodes, LADattr) = load_LAD_attributes()

rawlst = load_LAD_images(LADcodes)

(trainData, testData, trainAttr, testAttr, trainFiles, testFiles) = train_test_split(rawlst, LADattr, LADcodes, test_size = 0.10, random_state = 42)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu", padding = "same", input_shape = (484, 484, 1)))
model.add(Activation("relu")
model.add(BatchNormalization(axis = -1))

model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (9, 9), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))

model.add(MaxPooling2D(pool_size = (4, 4)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (7, 7), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))

model.add(MaxPooling2D(pool_size = (4, 4)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (5, 5), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))

model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(64, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))

model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(32, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(32, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))

model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(16, (5, 5), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(16, (5, 5), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(16, (5, 5), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(16, (5, 5), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))

model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(16))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.33))

model.add(Flatten())
model.add(Dense(8))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(4, activation = "sigmoid"))

sgd = SGD(learning_rate = 0.01, momentum = 0.9, nesterov = True)
model.compile(loss = "mse", optimizer = sgd, metrics = ["accuracy"])

trained = model.fit(trainData, trainAttr, validation_data = (testData, testAttr), batch_size = 16, epochs = 200, verbose = 1)

preds = model.predict(testData, batch_size = 16)
print(classification_report(testAttr.argmax(axis = 1), preds.argmax(axis = 1), target_names = ["LADLTX", "LADLTY", "LADRBX", "LADRBY"]))

model.save("LADDetector", save_format = "h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 200), trained.history["loss"], label = "train_loss")
plt.plot(np.arange(0, 200), trained.history["val_loss"], label="val_loss")
plt.title("LAD Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("LAD_Loss_Diagram.png")
