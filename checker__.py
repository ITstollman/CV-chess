# from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications.vgg19 import VGG19
# from tensorflow.keras.applications.vgg19 import preprocess_input
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from keras import Model
from keras.applications import VGG19
from keras.applications.convnext import preprocess_input, decode_predictions
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array

train_path = 'C://Users//itama//PycharmProjects//yael_checker//train'
test_path = 'C://Users//itama//PycharmProjects//yael_checker//test'
val_path = 'C://Users//itama//PycharmProjects//yael_checker//val'

x_train = []

for folder in os.listdir(train_path):

    sub_path = train_path + "/" + folder
    print(sub_path)

    print('NEW FOLDER', folder)
    for img in os.listdir(sub_path):
        image_path = sub_path + "/" + img

        print(image_path)
        img_arr = cv2.imread(image_path)

        img_arr = cv2.resize(img_arr, (224, 224))

        x_train.append(img_arr)
        print("HERE WE GO")
        print(x_train)

x_test = []

for folder in os.listdir(test_path):

    sub_path = test_path + "/" + folder

    for img in os.listdir(sub_path):
        image_path = sub_path + "/" + img

        img_arr = cv2.imread(image_path)

        img_arr = cv2.resize(img_arr, (224, 224))

        x_test.append(img_arr)

x_val = []

for folder in os.listdir(val_path):

    sub_path = val_path + "/" + folder

    for img in os.listdir(sub_path):
        image_path = sub_path + "/" + img

        img_arr = cv2.imread(image_path)

        img_arr = cv2.resize(img_arr, (224, 224))

        x_val.append(img_arr)

train_x = np.array(x_train)
test_x = np.array(x_test)
val_x = np.array(x_val)

print("AA", train_x)

train_x = train_x / 255.0
test_x = test_x / 255.0
val_x = val_x / 255.0

print("BB", train_x)

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

print("GG", train_datagen)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='sparse')
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='sparse')
val_set = val_datagen.flow_from_directory(val_path,
                                          target_size=(224, 224),
                                          batch_size=32,
                                          class_mode='sparse')

train_y = training_set.classes
test_y = test_set.classes
val_y = val_set.classes

print("zz", training_set.class_indices)
print("zz", test_set.class_indices)

# aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

vgg = VGG19(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)

# adding output layer.Softmax classifier is used as it is multi-class classification
prediction = Dense(12, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)

print(model.summary())

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
# Early stopping to avoid overfitting of model


# fit the model
history = model.fit(
    train_x,
    train_y,
    validation_data=(val_x, val_y),
    epochs=10,
    callbacks=[early_stop],
    batch_size=32, shuffle=True)

import matplotlib

matplotlib.use('TkAgg')

plt.plot(history.history['accuracy'], label='train acc')

model.evaluate(test_x, test_y, batch_size=32)

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(test_x)
y_pred = np.argmax(y_pred, axis=1)
# get classification report
print(classification_report(y_pred, test_y))
# get confusion matrix
print(confusion_matrix(y_pred, test_y))
# https://www.analyticsvidhya.com/blog/2021/07/step-by-step-guide-for-image-classification-on-custom-datasets/


# result = model('C:\\Users\\itama\\PycharmProjects\\yael_checker\\train\\48p\\596276.jpg')
# print('result', result)
#
# result = model('C:\\Users\\itama\\PycharmProjects\\yael_checker\\train\\49p\\124306.jpg')
# print('result', result)

# load an image from file
image = load_img('C:\\Users\\itama\\PycharmProjects\\yael_checker\\train\\48p\\596276.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model.predict(image)
# convert the probabilities to class labels
label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))


#Load flower jpeg image from local and set target size to 224 x 224
img = image.load_img('C:\\Users\\itama\\PycharmProjects\\yael_checker\\train\\48p\\596276.jpg', target_size=(224,224))


#convert image to array
input_img = image.img_to_array(img)
input_img = np.expand_dims(input_img, axis=0)

#convert image to array
input_img = image.img_to_array(img)
input_img = np.expand_dims(input_img, axis=0)
print('input_img', input_img)


#Predict the inputs on the model
predict_img = model.predict(input_img)
print('predict_img', predict_img)


# #Let's predict top 5 results
# top_five_predict = vgg16.decode_predictions(predict_img, top=5)
# top_five_predict