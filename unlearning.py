import joblib
from torch.utils.data import DataLoader

from eval import evaluate_unlearning
from eval import evaluate
from dataset import FMADataset
from model import Cnn6
from config import *
import time

def unlearning_main():
    start_time = time.time()
    # --- CARICA MODELLO E LABEL ENCODER ---
    model = Cnn6().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)) # carica i pesi salvati dall'addestramento
    #model.load_state_dict(torch.load("saved_models/fma_cnn_small_lr_0.0001.pth", map_location=DEVICE))  # carica i pesi salvati dall'addestramento

    model.eval()

    le = joblib.load(ENCODER_PATH)

    # carica gli split
    train_ids = joblib.load("joblib/train_ids.joblib")
    train_labels = joblib.load("joblib/train_labels.joblib")
    val_ids = joblib.load("joblib/val_ids.joblib")
    val_labels = joblib.load("joblib/val_labels.joblib")
    test_ids = joblib.load("joblib/test_ids.joblib")
    test_labels = joblib.load("joblib/test_labels.joblib")
    accuracy_train = joblib.load("joblib/accuracy_train.joblib")

    # FORGET GENRE
    forget_ids, forget_labels, retain_ids, retain_labels = forget_genre(train_ids, train_labels, le, genre_to_remove="Hip-Hop")

    #FORGET ARTIST
#    forget_ids, forget_labels, retain_ids, retain_labels = forget_artist(train_ids, train_labels)

    retain_dataset = FMADataset(retain_ids, retain_labels)
    forget_dataset = FMADataset(forget_ids, forget_labels)
    val_dataset = FMADataset(val_ids, val_labels)

    retain_loader = DataLoader(retain_dataset, batch_size=BATCH_SIZE, shuffle=False)
    forget_loader = DataLoader(forget_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- OTTIMIZZATORE E LOSS ---
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- applica algoritmo di UNLEARNING ---
    #unl_fine_tuning(model, forget_loader, criterion, optimizer)
    unl_gradient_ascent(model, forget_loader, retain_loader, criterion, optimizer)

    # --- SALVA MODELLO AGGIORNATO ---
    torch.save(model.state_dict(), UNL_MODEL_PATH)
    print(f"Modello aggiornato salvato in {UNL_MODEL_PATH}")

    # --- evaluate ---
    evaluate_unlearning(model, forget_loader, retain_loader, val_loader, accuracy_train)
    evaluate(model, val_loader, le)
    print(f"Tempo Unlearning: {time.time() - start_time:.2f} s")


def unl_fine_tuning(model, forget_loader, criterion, optimizer):
    """
    Fine-tuning inverso sui dati da dimenticare.
    """
    model.train()
    for epoch in range(UNL_EPOCHS):
        for inputs, labels in forget_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs['clipwise_output']
            loss = -criterion(outputs, labels)  # Loss negativa per far dimenticare
            loss.backward()
            optimizer.step()

    print(f"Complete {UNL_EPOCHS} of UNLEARNING con FINE TUNING")

def unl_gradient_ascent(model, forget_loader, retain_loader, criterion, optimizer, alpha=0.1, beta=0.9):
    """
    Algoritmo di unlearning più complesso:
    - Usa gradient ascent sui dati da dimenticare.
    - Usa gradient descent sui dati da mantenere (regolarizzazione).
    - Controlla il bilanciamento tramite i pesi alpha e beta.
    """

    model.train()
    for epoch in range(UNL_EPOCHS):
        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)

        for _ in range(min(len(forget_loader), len(retain_loader))):
            # --- Batch da dimenticare ---
            try:
                f_inputs, f_labels = next(forget_iter)
                f_inputs, f_labels = f_inputs.to(DEVICE), f_labels.to(DEVICE)
            except StopIteration:
                break

            # --- Batch da mantenere ---
            try:
                r_inputs, r_labels = next(retain_iter)
                r_inputs, r_labels = r_inputs.to(DEVICE), r_labels.to(DEVICE)
            except StopIteration:
                break

            optimizer.zero_grad()

            # Forward su dati da dimenticare
            f_outputs = model(f_inputs)['clipwise_output']
            f_loss = criterion(f_outputs, f_labels)

            # Forward su dati da mantenere
            r_outputs = model(r_inputs)['clipwise_output']
            r_loss = criterion(r_outputs, r_labels)

            # Loss combinata:
            #   -alpha * f_loss → ascent (disimparare)
            #   +beta * r_loss → descent (preservare conoscenza utile)
            loss = -alpha * f_loss + beta * r_loss
            loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch+1}/{UNL_EPOCHS}] Forget Loss: {f_loss.item():.4f} | Retain Loss: {r_loss.item():.4f}")
    print(f"Complete {UNL_EPOCHS} of UNLEARNING con GRADIENT ASCENT")

def forget_genre(train_ids, train_labels, le, genre_to_remove="Hip-Hop"):

    idx_to_remove = le.transform([genre_to_remove])[0]
    print(f"Rimuovere il genere '{genre_to_remove}' (indice {idx_to_remove})")

    # Filtra i dati
    forget_ids, forget_labels, retain_ids, retain_labels = [], [], [], []

    for tid, label in zip(train_ids, train_labels):
        if label == idx_to_remove:
            forget_ids.append(tid)
            forget_labels.append(label)
        else:
            retain_ids.append(tid)
            retain_labels.append(label)

    return forget_ids, forget_labels, retain_ids, retain_labels

def forget_artist():
    a = 1


unlearning_main()