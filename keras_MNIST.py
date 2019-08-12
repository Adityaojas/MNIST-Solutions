import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

train = pd.read_csv('MNIST/train.csv').values
X_train = train[:, 1:] / 255.0
X_train = X_train.astype('float32')
y_train = train[:, 0]


X_test = pd.read_csv('MNIST/test.csv').values.astype('float32') / 255.0

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)

X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size = 0.25, random_state = 111)

model = Sequential()
model.add(Dense(256, input_shape=(784,),activation ='sigmoid'))
model.add(Dense(128, activation = 'sigmoid'))
model.add(Dense(10, activation = 'softmax'))

sgd = SGD(1e-2)
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
H = model.fit(X_t, y_t, validation_data = (X_v, y_v), epochs=100, batch_size = 128)
y_v_pred = model.predict(X_v, batch_size = 128)

print(classification_report(y_v.argmax(axis=1), y_v_pred.argmax(axis=1)))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label = 'train_loss')
plt.plot(np.arange(0, 100), H.history['acc'], label = 'train_acc')
plt.plot(np.arange(0, 100), H.history['val_loss'], label = 'val_loss')
plt.plot(np.arange(0, 100), H.history['val_acc'], label = 'val_acc')
plt.legend()
plt.savefig('MNIST/keras/keras_MNIST.png')
plt.show()


# fitting whole training data

model.fit(X_train, y_train, epochs = 100, batch_size = 128)
y_pred = model.predict(X_test, batch_size = 128)
y_pred = y_pred.argmax(axis=1)

ind = np.arange(1, y_pred.shape[0]+1)
df = pd.DataFrame({'ImageId': ind, 'Label': y_pred})
df.to_csv('MNIST/keras_FC/predictions.csv', index = False)







