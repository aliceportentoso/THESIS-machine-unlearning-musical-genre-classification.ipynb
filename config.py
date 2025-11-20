from dataclasses import dataclass, field
from datetime import datetime

DEVICE = "cuda"

@dataclass
class Config:

    SUBSET = "medium"

    # Learning
    LR = 0.0001
    MAX_EPOCHS = 100
    GENRE_TO_REMOVE = None

    # Unlearning
    UNL_METHOD = "FT"   # FT, GA, ST, OSM, A
    UNL_EPOCHS = 4
    GENRE_TO_FORGET = "Prova"

    if SUBSET == "medium":
        GENRES = ["Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock" ]
        MODEL_PATH = "saved_models/20251105-1251_LEARN_LR-0.0005_subset-medium_epochs-500_model.pth"

    else:
        GENRES = [ "Blues", "Classical", "Country", "Easy Listening", "Electronic", "Experimental", "Folk", "Hip-Hop"  ]
        MODEL_PATH = "saved_models/model_learning_20251022-1824_remove-None_epochs-200.pth"

    AUDIO_DIR = "fma_large"
    CSV_FILE = "fma_metadata/tracks.csv"
    ENCODER_PATH = f"label_encoder_{SUBSET}.joblib"
    SPLITS_DIR = f"data_splits/{SUBSET}-dataset_remove-{GENRE_TO_REMOVE}"

    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M"))

    # Audio parameters
    SAMPLE_RATE = 22050
    WINDOW_SIZE = 1024
    HOP_SIZE = 512
    N_MELS = 64
    NUM_CLASSES = 8
    NUM_FRAMES = 1292
    DURATION = 30
    BATCH_SIZE = 32
    fmin = 0
    fmax = SAMPLE_RATE // 2

    # System
    NUM_WORKERS = 8

    @classmethod
    def unl_name_path(cls):
        return (
            f"{cls.UNL_METHOD}-{cls.SUBSET}-LR_{cls.LR}-"
            f"epoch_{cls.UNL_EPOCHS}/forget-{cls.GENRE_TO_FORGET}"
        )

    @classmethod
    def name_path(cls):
        if cls.GENRE_TO_REMOVE is None:
            return (
                f"{cls.timestamp}_LEARN_LR-{cls.LR}_SUBSET-{cls.SUBSET}_"
                f"epochs-{cls.MAX_EPOCHS}"
            )
        else:
            return (
                f"{cls.timestamp}_LEARN_LR-{cls.LR}_SUBSET-{cls.SUBSET}_"
                f"remove-{cls.GENRE_TO_REMOVE}_epochs-{cls.MAX_EPOCHS}"
            )

    @classmethod
    def print_config(cls):
        print("---- TRAINING CONFIG ----")
        print(f"Epochs         : {cls.MAX_EPOCHS}")
        print(f"Learning rate  : {cls.LR}")
        print(f"Dataset SUBSET : {cls.SUBSET}")
        print(f"DEVICE         : {DEVICE}")
        print(f"Num classes    : {cls.NUM_CLASSES}")

    @classmethod
    def print_config_unl(cls):
        print("---- UNLEARNING CONFIG ----")
        print(f"Epochs         : {cls.UNL_EPOCHS}")
        print(f"Learning rate  : {cls.LR}")
        print(f"Dataset SUBSET : {cls.SUBSET}")
        print(f"NAME           : {cls.unl_name_path()}")
