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