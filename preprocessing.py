import numpy as np
import os
import keras

from sklearn.model_selection import train_test_split
import nibabel as nib
from keras.utils import to_categorical
import config
from utils import extract_patches, augment_data
from sklearn.model_selection import train_test_split
def normalize(volume):
    """Normalize the volume"""
    min = np.min(volume)
    max = np.max(volume)
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def one_hot_encode(labels):
    """One-hot encode the labels"""
    return to_categorical(labels, num_classes=config.NUM_CLASSES)

def load_data(data_dir):
    data = []
    labels = []

    for subject_folder in os.listdir(data_dir):
        if subject_folder.startswith("subject-"):
            subject_id = subject_folder.split("subject-")[1]
            subject_dir = os.path.join(data_dir, subject_folder)

            # Load T1 data
            t1_file = f"subject-{subject_id}-T1.hdr"
            t1_path = os.path.join(subject_dir, t1_file)
            t1_volume = nib.load(t1_path).get_fdata()
            t1_volume = np.squeeze(t1_volume)  # Remove extra dimension
            t1_volume = np.expand_dims(t1_volume, axis=-1)
            
            
            print(t1_volume.shape)
            data.append(t1_volume)

            # Load T2 data
            t2_file = f"subject-{subject_id}-T2.hdr"
            t2_path = os.path.join(subject_dir, t2_file)
            t2_volume = nib.load(t2_path).get_fdata()
            t2_volume = np.squeeze(t2_volume)  # Remove extra dimension
            t2_volume = np.expand_dims(t2_volume, axis=-1)
            print(t2_volume.shape)
            data.append(t2_volume)

            # Load label
            label_file = f"subject-{subject_id}-label.hdr"
            label_path = os.path.join(subject_dir, label_file)
            label_volume = nib.load(label_path).get_fdata()
            label_volume = np.squeeze(label_volume)  # Remove extra dimension
            label_volume = np.expand_dims(label_volume, axis=-1)
            labels.append(label_volume)
            print(label_volume.shape)
            labels.append(label_volume)

    return np.array(data), np.array(labels)

def reshape_volume(volume):
    return np.expand_dims(volume, axis=0)







    

    # Perform any additional preprocessing steps or save the data as needed







def split_data(data, labels, validation_split, test_split, random_state):
    if len(data) != len(labels):
        raise ValueError("Number of samples in data and labels arrays do not match.")

    if len(data) < 2:
        raise ValueError("Insufficient number of samples for train-test split.")

    # Split data into train and remaining data
    X_train_val, X_test, y_train_val, y_test = train_test_split(data, labels, test_size=test_split, random_state=random_state)

    # Split remaining data into validation and test
    validation_size = validation_split / (1 - test_split)  # Adjust validation size based on remaining data
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=validation_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test




def preprocess_data():
    
    """Load, preprocess, and split data"""
    data, labels = load_data(config.DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, labels, config.VALIDATION_SPLIT, config.TEST_SPLIT, config.RANDOM_STATE)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    # Extract patches from the 3D volumes
    patch_size = (config.PATCH_SIZE, config.PATCH_SIZE, config.PATCH_SIZE)
    X_train, y_train = extract_patches(X_train, y_train, patch_size)
    patch_size = (config.PATCH_SIZE, config.PATCH_SIZE, config.PATCH_SIZE)
    X_val, y_val = extract_patches(X_val, y_val, patch_size)
    
    # Augment data
    X_train, y_train = next(augment_data(X_train, y_train, config.BATCH_SIZE, config.NUM_CLASSES))

    
    # Save preprocessed data
    np.save(os.path.join(config.DATA_PATH, 'X_train.npy'), X_train)
    np.save(os.path.join(config.DATA_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(config.DATA_PATH, 'X_val.npy'), X_val)
    np.save(os.path.join(config.DATA_PATH, 'y_val.npy'), y_val)
    np.save(os.path.join(config.DATA_PATH, 'X_test.npy'), X_test)
    np.save(os.path.join(config.DATA_PATH, 'y_test.npy'), y_test)

if __name__ == "__main__":
    keras.backend.set_image_data_format('channels_last')
    preprocess_data()
