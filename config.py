import torch
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d-%H%M")

LR = 0.0005
SUBSET = 'medium'

# LEARN
#REMOVE = None #GENRE, ARTIST, None
GENRE_TO_REMOVE = None #"Hip-Hop"
MAX_EPOCHS = 400 #100

if GENRE_TO_REMOVE is None:
    NAME = f'{timestamp}_LEARN_LR-{LR}_subset-{SUBSET}_epochs-{MAX_EPOCHS}'
else:
    NAME = f'{timestamp}_LEARN_LR-{LR}_subset-{SUBSET}_remove-{GENRE_TO_REMOVE}_epochs-{MAX_EPOCHS}'

MODEL_PATH = f'saved_models/{NAME}_model.pth'
ENCODER_PATH = f'label_encoder_{SUBSET}.joblib'

# UNLEARN
#TYPE_FORGET = None #GENRE, ARTIST, None
GENRE_TO_FORGET = "International"
UNL_EPOCHS = 2
UNL_NAME = f'{timestamp}_unl_forget-{GENRE_TO_FORGET}_epochs-{UNL_EPOCHS}'
UNL_MODEL_PATH = f'saved_models/{UNL_NAME}_unl_model.pth'

LEARN_MODEL_PATH = 'saved_models/model_learning_20251022-1824_remove-None_epochs-200.pth'

# --- CONFIG ---
SAMPLE_RATE = 22050
WINDOW_SIZE = 1024
HOP_SIZE = 512
N_MELS = 64
fmin = 0
fmax = SAMPLE_RATE // 2
if SUBSET == 'small':
    NUM_CLASSES = 8
if SUBSET == 'medium':
    NUM_CLASSES = 8
if SUBSET == 'large':
    NUM_CLASSES = 8

NUM_FRAMES = 1292

DURATION = 30
BATCH_SIZE = 32 # 32 -> (accuracy 2.14)      36 -> (accuracy 0.22)


AUDIO_DIR = f'fma_large'
CSV_FILE = 'fma_metadata/tracks.csv'
SPLITS_DIR = f"data_splits/{SUBSET}-dataset_remove-{GENRE_TO_REMOVE}"

NUM_WORKERS = 8 # 4 per curie, 8 per dirac
DEVICE = torch.device("cuda")

def print_config():
    print("---- TRAINING CONFIG ----")
    print(f"Epochs         : {MAX_EPOCHS}")
    print(f"Learning rate  : {LR}")
    print(f"Dataset subset : {SUBSET}")
    print(f"Device         : {DEVICE}")
    print(f"Num classes    : {NUM_CLASSES}")