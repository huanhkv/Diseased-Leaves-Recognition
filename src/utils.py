import os
import random

import cv2
import numpy as np
import tensorflow as tf

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

def readimg(filepath, image_size=(256, 256)):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = image / 255.0
    
    return image

def save_dataset(folderpath, x_train, y_train, x_valid, y_valid, x_test, y_test):
    print('Saving preprocessing dataset...')

    np.save(f'{folderpath}x_train.npy', x_train)
    np.save(f'{folderpath}y_train.npy', y_train)
    np.save(f'{folderpath}x_valid.npy', x_valid)
    np.save(f'{folderpath}y_valid.npy', y_valid)
    np.save(f'{folderpath}x_test.npy', x_test)
    np.save(f'{folderpath}y_test.npy', y_test)

def load_dataset(folderpath):
    print('Loading preprocessing dataset...')

    x_train = np.load(f'{folderpath}x_train.npy')
    y_train = np.load(f'{folderpath}y_train.npy')
    x_valid = np.load(f'{folderpath}x_valid.npy')
    y_valid = np.load(f'{folderpath}y_valid.npy')
    x_test = np.load(f'{folderpath}x_test.npy')
    y_test = np.load(f'{folderpath}y_test.npy')

    return x_train, y_train, x_valid, y_valid, x_test, y_test