import os, shutil, glob
import random
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image

base_data_dir = './data/preprocessed/'
train_dir = './data/bastard_train/'
val_dir = './data/bastard_val/'
test_dir = './data/bastard_test/'

size = 150

def remove_files(dir):
    files = glob.glob(dir + '*.jpg')
    for f in files:
        os.remove(f)

def get_files(dir):
    all_files = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        all_files.extend(filenames)
    
    return all_files

def split_files():
    sick_files = get_files(base_data_dir + 'sick/')
    healthy_files = get_files(base_data_dir + 'healthy/')

    sick_train_set_size = int(len(sick_files)*0.8)
    sick_val_set_size = int(len(sick_files)*0.1)

    healthy_train_set_size = int(len(healthy_files)*0.8)
    healthy_val_set_size = int(len(healthy_files)*0.1)

    for i in range(len(sick_files)):
        dest = ''
        if i < sick_train_set_size:
            dest = train_dir + 'sick/'
        elif i >= sick_train_set_size and i < sick_train_set_size + sick_val_set_size:
            dest = val_dir + 'sick/'
        elif i >= sick_train_set_size + sick_val_set_size:
            dest = test_dir + 'sick/'
    
        shutil.copyfile(base_data_dir+ 'sick/' + sick_files[i], dest + sick_files[i])

    for i in range(len(healthy_files)):
        dest = ''
        if i < healthy_train_set_size:
            dest = train_dir + 'healthy/'
        elif i >= healthy_train_set_size and i < healthy_train_set_size + healthy_val_set_size:
            dest = val_dir + 'healthy/'
        elif i >= healthy_train_set_size + healthy_val_set_size:
            dest = test_dir + 'healthy/'
    
        shutil.copyfile(base_data_dir+ 'healthy/' + healthy_files[i], dest + healthy_files[i])


def create_model():
    model = Sequential()
    model.add(Conv2D((16), (3,3), activation='relu', input_shape=(size, size, 3)))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D((32), (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D((64), (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D((128), (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D((128), (3,3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def load_test_data():
    sick_files = glob.glob(test_dir+ 'sick/*.jpg')
    healthy_files = glob.glob(test_dir+ 'healthy/*.jpg')
    length = len(sick_files) + len(healthy_files)

    x = np.zeros((length, size, size, 3), dtype='float')
    y = np.zeros(length, dtype='float')

    for i in range(len(sick_files)):
        y[i] = 1
        img = image.load_img(sick_files[i], target_size=(size, size))
        x[i] = image.img_to_array(img)

    for i in range(len(healthy_files)):
        img = image.load_img(healthy_files[i], target_size=(size, size))
        x[i + len(sick_files)] = image.img_to_array(img)

    return x, y


remove_files(train_dir + 'sick/')
remove_files(val_dir + 'sick/')
remove_files(test_dir + 'sick/')
remove_files(train_dir + 'healthy/')
remove_files(val_dir + 'healthy/')
remove_files(test_dir + 'healthy/')

split_files()

model = create_model()

train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2)
test_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.15, height_shift_range=0.15, zoom_range=0.2)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(size, size), batch_size=2, class_mode='binary')
val_generator = test_datagen.flow_from_directory(val_dir, target_size=(size, size), batch_size=2, class_mode='binary')

cb = [ModelCheckpoint('bastard.h5', monitor='val_acc', save_best_only=True)]

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100 , validation_data=val_generator, validation_steps=50, callbacks=cb)

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(acc) +1)
plt.plot(epochs, acc, 'bo', label='acc')
plt.plot(epochs, val_acc, 'b', label='val_acc')
plt.legend()
plt.savefig('bastard_acc.png')

test_x, test_y = load_test_data()
model = load_model('bastard.h5')
result = model.evaluate(test_x, test_y, batch_size=64)
prediction = model.predict(test_x)
print(result)
print(prediction)
