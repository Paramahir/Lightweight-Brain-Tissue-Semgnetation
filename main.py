from utils import DataGenerator
from train import train_model
from model import create_model
from preprocess import preprocess_data
import config
import os

def main():
    # Preprocess and save data as numpy arrays
    preprocess_data()

    # Create lists of training and validation IDs
    train_IDs = os.listdir(config.TRAIN_DATA_PATH)
    validation_IDs = os.listdir(config.VALIDATION_DATA_PATH)

    # Create data generators
    training_generator = DataGenerator(train_IDs)
    validation_generator = DataGenerator(validation_IDs)

    # Create model
    model = create_model()

    # Train model
    train_model(model, training_generator, validation_generator)

if __name__ == "__main__":
    main()
