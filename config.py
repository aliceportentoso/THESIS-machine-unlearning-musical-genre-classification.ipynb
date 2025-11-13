import torch
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d-%H%M")

LR = 0.0001
SUBSET = 'medium'

# LEARN
GENRE_TO_REMOVE = None
MAX_EPOCHS = 1

if GENRE_TO_REMOVE is None:
    NAME = f'{timestamp}_LEARN_LR-{LR}_subset-{SUBSET}_epochs-{MAX_EPOCHS}'
else:
    NAME = f'{timestamp}_LEARN_LR-{LR}_subset-{SUBSET}_remove-{GENRE_TO_REMOVE}_epochs-{MAX_EPOCHS}'

MODEL_PATH = f'saved_models/{NAME}_model.pth'
ENCODER_PATH = f'label_encoder_{SUBSET}.joblib'

# UNLEARN
GENRE_TO_FORGET = "Experimental" # "Blues", "Classical", "Country", "Easy Listening", "Electronic", "Experimental", "Folk", "Hip-Hop"
UNL_EPOCHS = 2 #15 per FT, 5 per GA
UNL_METHOD = "FT" # FT, GA, ST, OSM, A
UNL_NAME = f'METHOD_{UNL_METHOD}-genre_{GENRE_TO_FORGET}-{timestamp}' # unl-small-LR_5e-05-GA-4_epochs
UNL_MODEL_PATH = f'saved_models/{UNL_NAME}_unl_model.pth'

if SUBSET == 'small':
    LEARN_MODEL_PATH = 'saved_models/model_learning_20251022-1824_remove-None_epochs-200.pth'
else:
    LEARN_MODEL_PATH = 'saved_models/20251105-1251_LEARN_LR-0.0005_subset-medium_epochs-500_model.pth'
                        #'20251102-0820_LEARN_LR-0.0005_subset-medium_epochs-400_model.pth')

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
BATCH_SIZE = 32

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

def print_config_unl():
    print("---- UNLEARNING CONFIG ----")
    print(f"Epochs         : {UNL_EPOCHS}")
    print(f"Learning rate  : {LR}")
    print(f"Dataset subset : {SUBSET}")
    print(f"NAME           : {UNL_NAME}")