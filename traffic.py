import time
import cv2
import os
import sys
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import tensorflow.keras as tfk
from tensorflow.python.keras.callbacks import TensorBoard

EPOCHS = 15
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4
batch_size = 16
pool_size = (2, 2)
inputShape = (IMG_WIDTH, IMG_HEIGHT, 3)
Name = "trafficsSignsModel-{}".format(int(time.time()))


def main():
	# Check command-line arguments
	if len(sys.argv) not in [1, 3]:
		sys.exit("Usage: python traffic.py data_directory [model.h5]")

	# Get image arrays and labels for all image files
	images, labels = load_data(os.path.dirname(sys.argv[0]))


	# Split data into training and testing sets
	labels = tfk.utils.to_categorical(labels)
	x_train, x_test, y_train, y_test = train_test_split(
		np.array(images), np.array(labels), test_size=TEST_SIZE)

	# Get a compiled neural network
	model = get_model()
	model.summary()
	# Fit model on training data
	tensorboard = TensorBoard(log_dir='logs/{}'.format(Name))

	history = model.fit(x_train, y_train, epochs=EPOCHS,callbacks=[tensorboard])

	# Evaluate neural network performance
	_,acc_train =model.evaluate(x_train, y_train, verbose=2)
	_,acc_test = model.evaluate(x_test,y_test,verbose=2)
	print('Train accuracy : %.3f, Test accuracy: %.3f' % (acc_train, acc_test))

	# Save model to file
	if len(sys.argv) == 1:
		folder_name = os.path.dirname(sys.argv[0])
		print(os.path.join(folder_name,"model1.h5"))
		model.save(os.path.join(folder_name,"model1.h5"))
		print(f"Model saved to {folder_name}.")
		plt.figure(0)
		plt.plot(history.history['loss'], label='training loss')
		plt.title("Loss")
		plt.xlabel("Epochs")
		plt.ylabel("loss")
		plt.legend()
		plt.show()

		plt.figure(1)
		plt.plot(history.history['accuracy'], label='training accuracy ')
		plt.title("accuracy")
		plt.xlabel("Epochs")
		plt.ylabel("accu")
		plt.legend()
		plt.show()


def load_data(data_dir):
	data = []
	labels = []
	for i in range(NUM_CATEGORIES):
		path = os.path.join(data_dir, "gtsrb", str(i))
		images = os.listdir(path)
		for j in images:
			try:
				image = cv2.imread(os.path.join(path, j))
				image_from_array = Image.fromarray(image, 'RGB')
				resized_image = image_from_array.resize((IMG_HEIGHT, IMG_WIDTH))
				data.append(np.array(resized_image))
				labels.append(i)
			except AttributeError:
				print("Error loading the image!")

	images_data = (data, labels)
	return images_data


def get_model():
	# initialize the model along with the input shape to be
	# "channels last" and the channels dimension itself
	model = Sequential()
	chDimension = -1
	# 1 set of layers (CONV , RELU , BNormalization and POOL layers)
	model.add(Conv2D(8, (5, 5), padding="same",input_shape=inputShape))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chDimension))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# first 2 sets of layers 1st set (CONV , RELU ,) and another one of( CONV and RELU) then POOL layer
	model.add(Conv2D(16, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chDimension))
	model.add(Conv2D(16, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chDimension))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# second 2 second two sets of (CONV , RELU ,CONV  RELU) layers and POOL layer
	model.add(Conv2D(32, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chDimension))
	model.add(Conv2D(32, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chDimension))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# first set of fully connected layers , RELU layers
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation("relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.7))
	# second set of fully connected layers ,RELU layers
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation("relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.7))
	# softmax classifier ,The softmax function is used as the activation function in the output layer of neural network models that predict a multinomial probability distribution.
	# That is, softmax is used as the activation function for multi-class classification problems where class membership is required on more than two class labels.
	model.add(Dense(NUM_CATEGORIES))
	model.add(Activation("softmax"))

	# compiling the model
	model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=["accuracy"])

	return model


if __name__ == "__main__":
	main()