


train_path = 'C://Users//itama//PycharmProjects//yael_checker//train'
test_path = 'C://Users//itama//PycharmProjects//yael_checker//test'
val_path = 'C://Users//itama//PycharmProjects//yael_checker//val'

x_train = []

for folder in os.listdir(train_path):
    sub_path = train_path + "/" + folder
    for img in os.listdir(sub_path):
        print(img)
        image_path = sub_path + "/" + img
        img_arr = cv2.imread(image_path)
        img_arr = cv2.resize(img_arr, (224, 224))
        x_train.append(img_arr)

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

print(train_x.shape, test_x.shape, val_x.shape)

train_x = train_x / 255.0
test_x = test_x / 255.0
val_x = val_x / 255.0

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

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

print(training_set.class_indices)

train_y = training_set.classes
test_y = test_set.classes
val_y = val_set.classes
print(train_y.shape, test_y.shape, val_y.shape)