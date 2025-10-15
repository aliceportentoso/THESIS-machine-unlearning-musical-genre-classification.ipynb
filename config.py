import torch

# --- CONFIG ---
SAMPLE_RATE = 22050
WINDOW_SIZE = 1024
HOP_SIZE = 512
N_MELS = 64
fmin = 0
fmax = SAMPLE_RATE // 2
NUM_CLASSES = 8 # 8 per small, 16 per medium, 161 per large
NUM_FORGET = 1

DURATION = 30
BATCH_SIZE = 32 #meno per small
LR = 0.0001

MAX_EPOCHS = 60
UNL_EPOCHS = 20

AUDIO_DIR, CSV_FILE = 'fma_large', 'fma_metadata/tracks.csv'
SUBSET = 'small'
NUM_WORKERS = 4 # 4 per small, o 8
DEVICE = torch.device("cuda")
ENCODER_PATH = 'joblib/label_encoder.joblib'

MODEL_PATH = f'saved_models/fma_cnn_{SUBSET}_lr_{LR}.pth'
UNL_MODEL_PATH = f'saved_models/unlearning_{SUBSET}_lr_{LR}.pth'

NUM_FRAMES = 1292  # esempio calcolato dal dataset

def print_config():
    print("---- TRAINING CONFIG ----")
    print(f"Batch size     : {BATCH_SIZE}")
    print(f"Epochs         : {MAX_EPOCHS}")
    print(f"Learning rate  : {LR}")
    print(f"Dataset subset : {SUBSET}")
    print(f"Device         : {DEVICE}")
    print(f"Num Workers    : {NUM_WORKERS}")
    print(f"Hop Size       : {HOP_SIZE}")
    print(f"Num Mels       : {N_MELS}")
    print(f"Num forget     : {NUM_FORGET}   ")

