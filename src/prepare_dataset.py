import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils import readimg, save_dataset

tqdm.pandas()

REPLICAS = 1
class Parameters:
    height = 256
    width = 256
    image_size = (height, width)
    test_size_split = 0.2
    train_bs = max(32, 8*REPLICAS)
    valid_bs = max(train_bs*2, 8*REPLICAS)
    dropout = 0.5
    epochs = 100
    n_leaf = 16

def load_extrernal_dataset_csv(extrernal_data_folderpath):

    print('Loading extrernal data...')
    
    extrernal_df = pd.read_csv(extrernal_data_folderpath + 'train.csv')
    extrernal_df['filepath'] = extrernal_df.image_id.apply(
        lambda x: f'{extrernal_data_folderpath}images/{x}.jpg'
    )

    print('Shape of extrernal dataset:', extrernal_df.shape)
    print()

    return extrernal_df

def load_raw_dataset_csv(data_folder):

    print('Loading leaf disease recognition dataset...')

    df = pd.read_csv(data_folder + 'all_dataset.csv')
    df['filepath'] = df.filename.apply(lambda filename: data_folder + 'images/' + filename)

    print('Shape of eaf disease recognition dataset:', df.shape)
    print()
    
    return df

def load_image_dataset(extrernal_df, 
                       train_df, 
                       valid_df, 
                       test_df):
    print('Loading dataset...')

    all_train_filepath = extrernal_df.filepath.to_list() + train_df.filepath.to_list()
    all_train_label = extrernal_df.healthy.to_list() + train_df.label.to_list()

    x_train = np.array([readimg(filepath) for filepath in tqdm(all_train_filepath)])
    y_train = np.array(all_train_label).reshape(-1, 1)

    x_valid = np.stack(valid_df.filepath.progress_apply(readimg))
    y_valid = valid_df.label.to_numpy().reshape(-1, 1)

    x_test = np.stack(test_df.filepath.progress_apply(readimg))
    y_test = test_df.label.to_numpy().reshape(-1, 1)

    print('Train set:', x_train.shape, y_train.shape)
    print('Valid set:', x_valid.shape, y_valid.shape)
    print('Test set: ', x_test.shape, y_test.shape)
    print()

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def main():
    args = Parameters()

    # Loading extrernal dataset
    extrernal_data_folderpath = '/content/plant-pathology/'
    extrernal_df = load_extrernal_dataset_csv(extrernal_data_folderpath)

    # Loading leaf disease recognition dataset...
    data_folder = '/content/drive/MyDrive/Colab Notebooks/[Vision]Diseased leaves recognition/data/raw/Leaf-Disease-Recognition-Dataset/'
    df = load_raw_dataset_csv(data_folder)

    # Train-test-split
    train_df, test_df = train_test_split(df, 
                                         test_size=args.test_size_split, 
                                         random_state=8, 
                                         stratify=df['label'])

    test_df, valid_df = train_test_split(test_df, 
                                         test_size=0.5, 
                                         random_state=8, 
                                         stratify=test_df['label'])

    print('Train set:', train_df.shape)
    print('Valid set:', valid_df.shape)
    print('Test set: ', test_df.shape)
    print()

    # 
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_image_dataset(extrernal_df, 
                                                                            train_df, 
                                                                            valid_df, 
                                                                            test_df)

    #
    folderpath_save = ''
    save_dataset(folderpath_save, x_train, y_train, x_valid, y_valid, x_test, y_test)

if __name__ == "__main__":
    main()