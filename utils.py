import numpy as np
import nibabel as nib
import config
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.util import view_as_windows

def load_nifti_file(filepath):
    """Load and return the Nifti file data."""
    nifti = nib.load(filepath)
    volume = nifti.get_fdata()
    return volume

def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = config.PATCH_SIZE
    desired_width = config.PATCH_SIZE
    desired_height = config.PATCH_SIZE
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def preprocess_data(X_data, y_data):
    """Preprocess the data by resizing and normalization"""
    X_data = np.array([resize_volume(x) for x in X_data])
    y_data = np.array([resize_volume(x) for x in y_data])
    X_data = normalize(X_data)
    return X_data, y_data

def train_val_split(X_data, y_data):
    """Split data into train/validation datasets"""
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=config.RANDOM_STATE)
    return X_train, X_val, y_train, y_val







def extract_patches(volume, label, patch_size):
    volume_shape = volume.shape
    label_shape = label.shape

    if len(volume_shape) != 5 or len(label_shape) != 5:
        raise ValueError("Invalid volume or label shape.")

    _,D, H, W, _ = volume_shape

    # Calculate the number of patches in each dimension
    num_patches_D = D // patch_size[0]
    num_patches_H = H // patch_size[1]
    num_patches_W = W // patch_size[2]

    # Calculate the remaining values after division
    rem_D = D % patch_size[0]
    rem_H = H % patch_size[1]
    rem_W = W % patch_size[2]

    # Pad volume and label if necessary
    pad_D = patch_size[0] - rem_D if rem_D > 0 else 0
    pad_H = patch_size[1] - rem_H if rem_H > 0 else 0
    pad_W = patch_size[2] - rem_W if rem_W > 0 else 0
    
    volume = np.pad(volume, ((0, 0), (0, pad_D), (0, pad_H), (0, pad_W), (0, 0)), mode='constant')
    label = np.pad(label, ((0, 0), (0, pad_D), (0, pad_H), (0, pad_W), (0, 0)), mode='constant')
   
    

    # Reshape volume and label to patches
    volume_patches = np.reshape(volume, (num_patches_D, patch_size[0], num_patches_H, patch_size[1], num_patches_W, patch_size[2], -1))
    label_patches = np.reshape(label, (num_patches_D, patch_size[0], num_patches_H, patch_size[1], num_patches_W, patch_size[2], -1))
    #label_patches = np.reshape(label, (-1, patch_size[0], patch_size[1], patch_size[2], 1))


    # Swap dimensions for compatibility with Keras conventions
    volume_patches = np.transpose(volume_patches, (0, 2, 4, 1, 3, 5, 6))
    label_patches = np.transpose(label_patches, (0, 2, 4, 1, 3, 5, 6))


    # Reshape to combine patches from all dimensions
    volume_patches = np.reshape(volume_patches, (-1, patch_size[0], patch_size[1], patch_size[2], volume_shape[-1]))
    label_patches = np.reshape(label_patches, (-1, patch_size[0], patch_size[1], patch_size[2], volume_shape[-1]))

    return volume_patches, label_patches








def augment_data(images, labels, batch_size, num_classes):
    seed = 1
    image_datagen = ImageDataGenerator(...)
    
    # Reshape input data to 4 dimensions
    images = np.reshape(images, (-1, images.shape[2], images.shape[3], images.shape[4]))
    labels = np.reshape(labels, (-1, labels.shape[2], labels.shape[3], labels.shape[4]))

    # Apply augmentation
    image_datagen.fit(images, augment=True, seed=seed)

    # Create an iterator
    data_generator = image_datagen.flow(images, labels, batch_size=batch_size, seed=seed)

    return data_generator



class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=config.BATCH_SIZE, dim=config.PATCH_SIZE, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 1))
        y = np.empty((self.batch_size, *self.dim, 1), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=config.NUM_CLASSES)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
