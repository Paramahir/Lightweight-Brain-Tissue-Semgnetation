import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import get_model
import config
import os
def load_preprocessed_data():
    """Load the preprocessed data"""
    X_train = np.load(os.path.join(config.DATA_PATH, 'X_train.npy'))
    y_train = np.load(os.path.join(config.DATA_PATH, 'y_train.npy'))
    X_val = np.load(os.path.join(config.DATA_PATH, 'X_val.npy'))
    y_val = np.load(os.path.join(config.DATA_PATH, 'y_val.npy'))
    return X_train, y_train, X_val, y_val

def compile_model(model):
    """Compile the model"""
    model.compile(optimizer=Adam(lr=config.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model with early stopping and model checkpoint callbacks"""
    resized_X_train = np.resize(X_train, (X_train.shape[0], 144, 192, 256, 1))
    resized_y_train = np.resize(y_train, (y_train.shape[0], 144, 192, 256, 1))
    resized_X_val = np.resize(X_val, (X_val.shape[0], 144, 192, 256, 1))
    resized_y_val = np.resize(y_val, (y_val.shape[0], 144, 192, 256, 1))
# Train the model with the resized input data
   
    callbacks = [
        EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, verbose=1),
        ModelCheckpoint(filepath=config.MODEL_SAVE_PATH, verbose=1, save_best_only=True)
    ]
    history = model.fit(
        x=resized_X_train,
        y=resized_y_train,
        validation_data=(resized_X_val, resized_y_val),
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        callbacks=callbacks
    )
    return history

if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_preprocessed_data()
    model = get_model((144, 192, 256, 1))
    model = compile_model(model)
    history = train_model(model, X_train, y_train, X_val, y_val)
