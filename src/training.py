import argparse
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from models import create_model
from prepare_dataset import load_dataset
from utils import seed_everything

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training model.')
    parser.add_argument('--processed_data', type=str, required=True, help='')
    parser.add_argument('--save_model', type=str, required=True, help='')
    parser.add_argument('--train_bs', type=int, default=32, help='')
    parser.add_argument('--valid_bs', type=int, default=64, help='')
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')

    return parser.parse_args()

def training(model,
             training_data, 
             batch_size, 
             epochs, 
             validation_data, 
             validation_batch_size, 
             callbacks, 
             return_history=True):

    # Warm-up
    print('Warm-up...')
    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[-4:0]:
        layer.trainable = True

    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(learning_rate=1e-3), 
                  metrics=["binary_accuracy"])

    model.fit(training_data[0], 
              training_data[1], 
              batch_size=32, 
              epochs=2, 
              shuffle=True)
    print()

    # Full model
    print('Traing...')
    for layer in model.layers:
        layer.trainable = True

    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(learning_rate=1e-3), 
                  metrics=["binary_accuracy"])

    history = model.fit(training_data[0], 
                        training_data[1],
                        batch_size=batch_size,
                        epochs=epochs, 
                        shuffle=True,
                        validation_data=(validation_data[0], validation_data[1]),
                        validation_batch_size=validation_batch_size,
                        callbacks=callbacks)

    if return_history:
        return model, history
    else:
        return model

def main():
    # Init
    seed = 99
    seed_everything(seed)
    args = parse_arguments()

    # Load data
    data_folderpath = args.processed_data
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_dataset(data_folderpath)
    
    # Create model
    input_shape = x_train.shape[1:]
    model = create_model(input_shape, dropout=args.dropout) 
    model.summary()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.save_model, 
                                           verbose=1,
                                           monitor='val_loss', 
                                           mode='min',
                                           save_best_only=True,
                                           save_weights_only=True)
    ]

    # Training
    model, history = training(model,
                              training_data=(x_train, y_train), 
                              batch_size=args.train_bs, 
                              epochs=args.epochs, 
                              validation_data=(x_valid, y_valid), 
                              validation_batch_size=args.valid_bs, 
                              callbacks=callbacks)

if __name__ == "__main__":
    main()

