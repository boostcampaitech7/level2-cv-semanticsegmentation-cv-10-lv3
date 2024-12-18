MODEL = "conv"
IMAGE_SIZE = 1024
FOLD = 0

RANDOM_SEED = 21

TRAIN_BATCH_SIZE = 2
VAL_BATCH_SIZE = 1

VAL_EVERY = 10
NUM_EPOCHS = 100
LR = 1e-4
TH = 0.5

SAVED_DIR = f"./checkpoints/result"
PT_NAME = f"{MODEL}_{IMAGE_SIZE}_{FOLD}.pt"
LOG_NAME = f"log_{MODEL}_{IMAGE_SIZE}_{FOLD}"

ALL_DATA = False
SLIDING = False

### for dataset
IMAGE_ROOT = "./data/train/DCM"
LABEL_ROOT = "./data/train/outputs_json"

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}