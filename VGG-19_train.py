import os
import cv2
import numpy as np
from keras import Model
from keras.applications import VGG19
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

IMAGE_SIZE = [224, 224]

paths = {'train': 'C://Users//itama//PycharmProjects//yael_checker//train',
         'test': 'C://Users//itama//PycharmProjects//yael_checker//test',
         'val': 'C://Users//itama//PycharmProjects//yael_checker//val'}

xs = [[], [], []]
ys = [[], [], []]
datagens = [[], [], []]
sets = [[], [], []]


#  going through the files
#  train -> classes -> images
#  test -> classes -> images
#  val -> classes -> images

for i, path in enumerate(paths.values()):
    for folder in os.listdir(path):
        sub_path = path + "/" + folder
        for img in os.listdir(sub_path):
            print(img)
            image_path = sub_path + "/" + img
            img_arr = cv2.imread(image_path)
            img_arr = cv2.resize(img_arr, (224, 224))
            xs[i].append(img_arr)

    xs[i] = np.array(xs[i])
    xs[i] = xs[i] / 255.0

    datagens[i] = ImageDataGenerator(rescale=1. / 255)
    sets[i] = datagens[i].flow_from_directory(path,
                                              target_size=(224, 224),
                                              batch_size=32,
                                              class_mode='sparse')
    ys[i] = sets[i].classes

vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)

prediction = Dense(13, activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
print(model.summary())

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# not allowing overfitting of the model using an early stop
history = model.fit(
    xs[0],
    ys[0],
    validation_data=(xs[2], ys[2]),
    epochs=10,
    callbacks=[early_stop],
    batch_size=32, shuffle=True)

# loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()

plt.savefig('vgg-loss-rps-1.png')
plt.show()

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()

plt.savefig('vgg-acc-rps-1.png')
plt.show()

model.evaluate(xs[1], ys[1], batch_size=32)
y_pred = model.predict(xs[1])
y_pred = np.argmax(y_pred, axis=1)

accuracy_score(y_pred, ys[1])

print(classification_report(y_pred, ys[1]))

print(classification_report(y_pred, ys[1]))
print(confusion_matrix(y_pred, ys[1]))

model.save("VGGMODEL.h5")




