from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from mymodule.nn.neuralnetwork import NeuralNetwork 
import numpy as np
import pandas as pd
from sklearn import datasets

digits = datasets.load_digits()
data = digits.data.astype('float') / 255.0
target = digits.target

lb = LabelBinarizer()
target = lb.fit_transform(target)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.25, random_state = 111)

layers = [X_train.shape[1], 32, 16, 10]
print('[INFO] loading model...')
model = NeuralNetwork(layers = layers)
print('[INFO] {}'.format(model))
print('[INFO] training...')
model.fit(X_train, y_train, 2500, 100)

y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)

print('\n [INFO] classification report:')

print(classification_report(y_test.argmax(axis=1), y_pred))
