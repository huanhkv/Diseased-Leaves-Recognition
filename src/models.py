import tensorflow as tf
from classification_models.tfkeras import Classifiers

from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def create_model(input_shape, dropout=0.5):
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    base = ResNet18(input_shape=input_shape, 
                    weights='imagenet', 
                    include_top=False
                   )
    
    x = GlobalAveragePooling2D(name='fc_global_average_pooling2d')(base.output)
    x = Dropout(dropout, name='fc_dropout1')(x)
    x = Dense(64, activation='relu', name='fc_dense')(x)
    x = Dropout(dropout, name='fc_dropout2')(x)
    output = Dense(1, activation='sigmoid', name='fc_output')(x)

    model = Model(base.input, output)
    return model

def load_model(model_filepath):
    pass

def save_model(model_filepath):
    pass