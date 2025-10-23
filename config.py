import torch
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d-%H%M")

LR = 0.00002

# LEARN
#REMOVE = None #GENRE, ARTIST, None
GENRE_TO_REMOVE = None #"Hip-Hop"
MAX_EPOCHS = 200
NAME = f'LR-{LR}_learning_{timestamp}_remove-{GENRE_TO_REMOVE}_epochs-{MAX_EPOCHS}'
MODEL_PATH = f'saved_models/model_{NAME}.pth'
ENCODER_PATH = 'label_encoder.joblib'

# UNLEARN
#TYPE_FORGET = None #GENRE, ARTIST, None
GENRE_TO_FORGET = 'Hip-Hop'
UNL_EPOCHS = 3
UNL_NAME = f'unlearning_{timestamp}_forget-{GENRE_TO_FORGET}_epochs-{UNL_EPOCHS}'
UNL_MODEL_PATH = f'saved_models/model_{UNL_NAME}.pth'

LEARN_MODEL_PATH = 'saved_models/model_learning_20251022-1824_remove-None_epochs-200.pth'

# --- CONFIG ---

SAMPLE_RATE = 22050
WINDOW_SIZE = 1024
HOP_SIZE = 512
N_MELS = 64
fmin = 0
fmax = SAMPLE_RATE // 2
NUM_CLASSES = 8 # 8 per small, 16 per medium, 161 per large
NUM_FRAMES = 1292

DURATION = 30
BATCH_SIZE = 32 # da aumentare per mediu,


SUBSET = 'small'
AUDIO_DIR = f'fma_{SUBSET}'
CSV_FILE = 'fma_metadata/tracks.csv'
SPLITS_DIR = f"data_splits/{SUBSET}-dataset_remove-{GENRE_TO_REMOVE}"

NUM_WORKERS = 4 # 4 per small, o 8
DEVICE = torch.device("cuda")

def print_config():
    print("---- TRAINING CONFIG ----")
    print(f"Epochs         : {MAX_EPOCHS}")
    print(f"Learning rate  : {LR}")
    print(f"Dataset subset : {SUBSET}")
    print(f"Device         : {DEVICE}")