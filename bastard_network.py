import os, shutil, glob
import random
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

base_data_dir = './data/preprocessed/'
train_dir = './data/bastard_train/'
val_dir = './data/bastard_val/'
test_dir = './data/bastard_test/'

def remove_files(dir):
    files = glob.glob(dir + '*.jpg')
    for f in files:
        os.remove(f)

def get_files(dir):
    all_files = []
    for (dirpath, dirnames, filenames) in os.walk(base_data_dir):
        all_files.extend(filenames)
    
    return all_files

def split_files():
    sick_files = get_files(base_data_dir + 'sick/')
    healthy_files = get_files(base_data_dir + 'healthy/')

    sick_train_set_size = int(len(sick_files)*0.8)
    sick_val_set_size = int(len(sick_files)*0.1)

    healthy_train_set_size = int(len(sick_files)*0.8)
    healthy_val_set_size = int(len(sick_files)*0.1)

    for i in range(len(sick_files) - 1):
        dest = ''
        if i < sick_train_set_size:
            dest = train_dir + 'sick/'
        elif i >= sick_train_set_size and i < sick_train_set_size + sick_val_set_size:
            dest = val_dir + 'sick/'
        elif i >= sick_train_set_size + sick_val_set_size:
            dest = test_dir + 'sick/'
    
        shutil.copyfile(base_data_dir+ 'sick/' + sick_files[i], dest + sick_files[i])

    for i in range(len(healthy_files) -1):
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
    model.add(Conv2D((16), (3,3), activation='relu', input_shape=(300, 300, 3)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

remove_files(train_dir + 'sick/')
remove_files(val_dir + 'sick/')
remove_files(test_dir + 'sick/')
remove_files(train_dir + 'healthy/')
remove_files(val_dir + 'healthy/')
remove_files(test_dir + 'healthy/')

split_files()

