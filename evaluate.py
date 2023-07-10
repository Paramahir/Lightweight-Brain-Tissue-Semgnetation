import numpy as np
from tensorflow.keras.models import load_model
import config

def load_preprocessed_test_data():
    """Load the preprocessed test data"""
    X_test = np.load(os.path.join(config.DATA_PATH, 'X_test.npy'))
    y_test = np.load(os.path.join(config.DATA_PATH, 'y_test.npy'))
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test data"""
    results = model.evaluate(X_test, y_test, batch_size=config.BATCH_SIZE)
    return results

if __name__ == "__main__":
    X_test, y_test = load_preprocessed_test_data()
    model = load_model(config.MODEL_PATH)
    results = evaluate_model(model, X_test, y_test)
    print(f"Test Loss: {results[0]}")
    print(f"Test Accuracy: {results[1]}")
