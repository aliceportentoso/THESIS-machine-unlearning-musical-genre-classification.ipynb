import numpy
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import time
from torchlibrosa.augmentation import SpecAugmentation

from dataset import FMADataset
from model import Cnn6
from train import train
from eval import evaluate
from config import *

print_config()
tracks = pd.read_csv(CSV_FILE,  index_col=0, header=[0,1])

#sub_tracks = tracks[tracks[('set', 'subset')].isin(["small", "medium"])] #25 mila
#sub_tracks = tracks[tracks[('set', 'subset')] == "small"]
sub_tracks = tracks[tracks[('set', 'subset')].isin(["small", "medium"])] #106 mila

track_genres = sub_tracks[('track', 'genre_top')].dropna() #49 mila
track_genres = track_genres.sort_values()

genres_to_exclude = ['Easy Listening', 'Blues', 'Soul-RnB', 'Country', 'Classical', 'Old-Time / Historic', 'Jazz', 'Spoken']

track_genres = track_genres[~track_genres.isin(genres_to_exclude)]
classes = genres_list = track_genres.unique().tolist()

# - elimina elementi per classi sovra rappresentate
MAX_PER_CLASS = 3000
print(f"max per class: {MAX_PER_CLASS}, NO augementer")
balanced_indices = []
for cls in classes:
    cls_indices = track_genres[track_genres == cls].index.tolist()
    if len(cls_indices) > MAX_PER_CLASS:
        cls_indices = list(numpy.random.choice(cls_indices, MAX_PER_CLASS, replace=False))
    balanced_indices.extend(cls_indices)
track_genres = track_genres.loc[balanced_indices]

le = LabelEncoder()
le.fit(classes)
joblib.dump(le, ENCODER_PATH) # -> non serve se è sempre lo stesso

if GENRE_TO_REMOVE is not None:

    print(f"Learning senza il genere {GENRE_TO_FORGET}..")
    genre_ids = sub_tracks[sub_tracks[('track', 'genre_top')] == GENRE_TO_FORGET].index #questo serve a non far eliminare completamente il genere
    track_genres = track_genres.drop(genre_ids, errors='ignore')

track_genres = track_genres.drop([1486,2624,3284,5574,8669,10116,11583,12838,13529,14116,14180,20814,22554,23429,23430,
                                  23431,25173,25174,25175,25176,25180,29345,29346,29352,29356,33411,33413,33414,33417,
                                  33418,33419,33425,35725,39363,41745,42986,43753,50594,50782,53668,54569,54582,61480,
                                  61822,63422,63997,65753,72656,72980,73510,80391,80553,82699,84503,84504,84522,84524,
                                  86656,86659,86661,86664,87057,90244,90245,90247,90248,90250,90252,90253,90442,90445,
                                  91206,92479,94052,94234,95253,96203,96207,96210,98105,98558,98559,98560,98562,98571,
                                  99134,101265,101272,101275,102241,102243,102247,102249,102289,105247,106409,106412,
                                  106415,106628,108920,108925,109266,110236,115610,117441,126981,127336,127928,129207,
                                  129800,130328,130748,130751,131545,133297,133641,133647,134887,140449,140450,140451,
                                  140452,140453,140454,140455,140456,140457,140458,140459,140460,140461,140462,140463,
                                  140464,140465,140466,140467,140468,140469,140470,140471,140472,142614,143992,144518,
                                  144619,145056,146056,147419,147424,148786,148787,148788,148789,148790,148791,148792,
                                  148793,148794,148795,151920,155051, 134956], errors='ignore') # dataset errors

track_ids = track_genres.index.values
labels = le.transform(track_genres)

train_ids, test_ids, train_labels, test_labels = train_test_split(
    track_ids, labels, test_size=0.2, random_state=42, stratify=labels
)

train_ids, val_ids, train_labels, val_labels = train_test_split(
    train_ids, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

# SAVE SPLITS
#if not os.path.isdir(SPLITS_DIR): # se la cartella non esiste la creo
#    os.makedirs(SPLITS_DIR, exist_ok=True)
joblib.dump(train_ids, f"{SPLITS_DIR}/train_ids.joblib")
joblib.dump(train_labels, f"{SPLITS_DIR}/train_labels.joblib")
joblib.dump(val_ids, f"{SPLITS_DIR}/val_ids.joblib")
joblib.dump(val_labels, f"{SPLITS_DIR}/val_labels.joblib")
joblib.dump(test_ids, f"{SPLITS_DIR}/test_ids.joblib")
joblib.dump(test_labels, f"{SPLITS_DIR}/test_labels.joblib")

augmenter = SpecAugmentation(time_drop_width=64,time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)

# CREATE DATASET AND ITERATORS
train_dataset = FMADataset(train_ids, train_labels, augmenter = None)
val_dataset   = FMADataset(val_ids, val_labels)
test_dataset  = FMADataset(test_ids, test_labels)
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# --- MODEL ---
model = Cnn6()
model.to(DEVICE)

# batch di dati e forward pass
batch_waveforms, batch_labels = next(iter(train_loader))
batch_waveforms = batch_waveforms.to(DEVICE)
batch_labels = batch_labels.to(DEVICE)
output_dict = model(batch_waveforms) #qui chiamo model-forward
clipwise_output = output_dict['clipwise_output']  # [batch_size, NUM_CLASSES]
embedding = output_dict['embedding']              # [batch_size, 512]
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- TRAINING ---
start_time = time.time()
train(model, train_loader, val_loader, criterion, optimizer, DEVICE)

# --- Evaluation and save model
accuracy = evaluate(model, test_loader, label_encoder=le)
#joblib.dump(accuracy, f"results/{NAME}_accuracy_train.joblib") #-> se non cambia l'ho già salvato. serve per fare confronto pre e post, per dump
torch.save(model.state_dict(), MODEL_PATH)

print_config()
print(f"Tempo Learning: {(time.time()-start_time)/3600:.2f} ore")
