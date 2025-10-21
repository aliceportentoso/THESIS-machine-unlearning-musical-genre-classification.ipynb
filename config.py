import torch

# --- CONFIG ---

TYPE_FORGET = "GENRE" #GENRE, ARTIST, None
MAX_EPOCHS = 200
UNL_EPOCHS = 2

MODEL_PATH = f'saved_models/epochs_{MAX_EPOCHS}.pth'
UNL_MODEL_PATH = f'saved_models/unlearning_epochs_{UNL_EPOCHS}.pth'

# -----

SAMPLE_RATE = 22050
WINDOW_SIZE = 1024
HOP_SIZE = 512
N_MELS = 64
fmin = 0
fmax = SAMPLE_RATE // 2
NUM_CLASSES = 8 # 8 per small, 16 per medium, 161 per large
NUM_FRAMES = 1292

DURATION = 30
BATCH_SIZE = 32 #meno per small
LR = 0.0001

AUDIO_DIR, CSV_FILE = 'fma_small', 'fma_metadata/tracks.csv'
SUBSET = 'small'
NUM_WORKERS = 4 # 4 per small, o 8
DEVICE = torch.device("cuda")
ENCODER_PATH = 'joblib/label_encoder.joblib'


def print_config():
    print("---- TRAINING CONFIG ----")
    print(f"Epochs         : {MAX_EPOCHS}")
    print(f"Learning rate  : {LR}")
    print(f"Dataset subset : {SUBSET}")
    print(f"Device         : {DEVICE}")
    print(f"Type Forget    : {TYPE_FORGET}   ")

