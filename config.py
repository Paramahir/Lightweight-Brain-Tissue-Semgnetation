# Configurations for iSeg2019 infant brain MRI segmentation project

# Path to the dataset
DATA_PATH = './iseg2019/'

# Names of the image files and mask files
IMAGE_FILENAMES = ['T1','T2']
MASK_FILENAMES = 'label'

# MRI scan modalities
MODALITIES = ['t1', 't2']

# Split proportions for training, validation and test sets
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Input image dimensions (we might be using patch of original volume due to GPU memory limitation)
PATCH_SIZE = 2

# Batch size for training
BATCH_SIZE = 1
RANDOM_STATE = 42


# Number of epochs for training
EPOCHS = 100

# Learning rate for optimizer
LEARNING_RATE = 0.0001

# Number of output classes
NUM_CLASSES = 4

# Early stopping patience
EARLY_STOPPING_PATIENCE = 10

# Model save path
MODEL_SAVE_PATH = './save_model/'

# Path to save metrics and evaluation results
RESULTS_PATH = './results/'
