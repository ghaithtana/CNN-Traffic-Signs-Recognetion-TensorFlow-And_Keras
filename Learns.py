import pickle
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from tensorflow import keras as tfk
import numpy as np
from traffic import load_data
from traffic import get_model
from traffic import EPOCHS,batch_size
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
import time,os,sys
from imblearn.over_sampling import SMOTE
from collections import Counter

Name = "trafficsSignsModel-{}".format(int(time.time()))
checkPointModel = "Model-{}.h5".format(int(time.time()))
modelName = "model{}.h5".format(int(time.time()))
model = get_model()
model.summary()
images, labels = load_data(os.path.dirname(sys.argv[0]))
labels = tfk.utils.to_categorical(labels)
x_train, x_test, y_train, y_test = train_test_split(
	np.array(images), np.array(labels), test_size=0.1,random_state=1)
with open('test.pkl', 'wb') as f:
	pickle.dump([x_test,y_test], f)


filepath ="Saved_models_checkpoints/{}".format(checkPointModel)
print(y_train.shape)
nsamples, npx,npy,rgb = x_train.shape

d2_train_x = x_train.reshape((nsamples,npx*npy*rgb))

smote = SMOTE()
X_train_smote, y_train_smote =smote.fit_sample(d2_train_x,y_train)

df = pd.DataFrame({"label":np.argmax(y_train_smote,axis=1)})
print(df['label'].value_counts())

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
M_checkP = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir='logs/{}'.format(Name))
# fit model
history = model.fit(x_train, y_train_smote, validation_split=0.22222, epochs=EPOCHS, verbose=1,callbacks=[es,M_checkP,tensorboard],batch_size=batch_size)
# evaluate the model
_, train_acc = model.evaluate(x_train, y_train, verbose=0)
_, test_acc = model.evaluate(x_test, y_test, verbose=0)
if len(sys.argv) == 1:
	folder_name = os.path.dirname(sys.argv[0])
	saved_model = model.save(os.path.join(folder_name, "model2.h5"))
	print(f"Model saved to {folder_name}.")
print('Train accuracy : %.3f, Test accuracy: %.3f' % (train_acc, test_acc))
# plot training history
pyplot.figure(0)
pyplot.plot(history.history['loss'], label='training loss')
pyplot.plot(history.history['val_loss'], label='validation loss')
pyplot.title("Loss")
pyplot.xlabel("Epochs")
pyplot.ylabel("loss")
pyplot.legend()
pyplot.show()

pyplot.figure(1)
pyplot.plot(history.history['accuracy'], label='training accuracy ')
pyplot.plot(history.history['val_accuracy'],label='validation accuracy')
pyplot.title("accuracy")
pyplot.xlabel("Epochs")
pyplot.ylabel("accu")
pyplot.legend()
pyplot.show()
