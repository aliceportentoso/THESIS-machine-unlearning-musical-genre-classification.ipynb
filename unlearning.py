import joblib
from eval import evaluate_unlearning
from dataset import FMADataset
from model import Cnn6
from config import *
import time
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.utils.data import DataLoader

def unlearning_main():
    retain_ids, forget_ids, retain_labels, forget_labels = [], [], [], []
    start_time = time.time()

    # --- CARICA MODELLO E LABEL ENCODER ---
    model = Cnn6().to(DEVICE)
    model.load_state_dict(torch.load(LEARN_MODEL_PATH, map_location=DEVICE)) # carica i pesi salvati dall'addestramento
    model.eval()
    le = joblib.load(ENCODER_PATH)

    # carica gli split
    dir_ = f"data_splits/{SUBSET}-dataset_remove-None"
    train_ids = joblib.load(f"{dir_}/train_ids.joblib")
    train_labels = joblib.load(f"{dir_}/train_labels.joblib")
    val_ids = joblib.load(f"{dir_}/val_ids.joblib")
    val_labels = joblib.load(f"{dir_}/val_labels.joblib")
    test_ids = joblib.load(f"{dir_}/test_ids.joblib")
    test_labels = joblib.load(f"{dir_}/test_labels.joblib")

    if GENRE_TO_FORGET is not None:
        forget_ids, forget_labels, retain_ids, retain_labels = forget_retain_split(train_ids, train_labels, le)

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
    unl_fine_tuning(model, forget_loader, criterion, optimizer)
    unl_gradient_ascent(model, forget_loader, retain_loader, criterion, optimizer)
    unl_stochastic_teacher(model, forget_loader, retain_loader, criterion, optimizer)
    unl_one_shot_magnitude(model, forget_loader, retain_loader, prune_ratio=0.2)
    unl_amnesiac(model, forget_loader, retain_loader, criterion=None, lr=1e-4, steps=1)

    # --- SALVA MODELLO AGGIORNATO ---
    torch.save(model.state_dict(), UNL_MODEL_PATH)
    print(f"Modello aggiornato salvato in {UNL_MODEL_PATH}")

    # --- evaluate ---
    evaluate_unlearning(model, forget_loader, retain_loader, val_loader, le)
    print(f"Tempo Unlearning: {(time.time() - start_time)/3600:.2f} ore")


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

def unl_gradient_ascent(model, forget_loader, retain_loader, criterion, optimizer, alpha=0.4, beta=0.6):
    """
    Algoritmo di unlearning più complesso:
    - Usa gradient ascent sui dati da dimenticare.
    - Usa gradient descent sui dati da mantenere (regolarizzazione).
    - Controlla il bilanciamento tramite i pesi alpha e beta.
    """
    losses = []

    model.train()
    for epoch in range(UNL_EPOCHS):
        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)
        epoch_loss = 0.0
        total_batches = 0

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

        epoch_loss += loss.item()
        total_batches += 1

        avg_loss = epoch_loss / total_batches
        losses.append(avg_loss)

        print(f"[Epoch {epoch + 1}/{UNL_EPOCHS}] Loss: {avg_loss:.4f}")
        print(f"[Epoch {epoch + 1}/{UNL_EPOCHS}] Forget Loss: {f_loss.item():.4f} | Retain Loss: {r_loss.item():.4f}")

    # --- Plot finale ---
    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker='o')
    plt.title("Loss")
    plt.show()
    plt.savefig(f"results/{UNL_NAME}_LOSS.png", bbox_inches='tight')  # bbox_inch

    print(f"Complete {UNL_EPOCHS} of UNLEARNING con GRADIENT ASCENT")

def unl_stochastic_teacher(model, forget_loader, retain_loader, criterion, optimizer, alpha=0.4, beta=0.6, randomize_labels=True):
    """
    Stochastic / Incompetent Teacher unlearning method.
    model: torch.nn.Module
    forget_loader: DataLoader (dati da dimenticare)
    retain_loader: DataLoader (dati da mantenere)
    criterion: funzione di perdita (es. nn.CrossEntropyLoss)
    optimizer: torch optimizer
    alpha, beta: pesi di bilanciamento
    device: 'cuda' o 'cpu'
    epochs: numero di epoche
    randomize_labels: se True, randomizza le label dei dati da dimenticare
    """

    model.train()
    model.to(DEVICE)

    for epoch in range(UNL_EPOCHS):
        retain_iter = iter(retain_loader)
        forget_iter = iter(forget_loader)

        num_batches = min(len(retain_iter), len(forget_iter))

        for _ in range(num_batches):
            try:
                x_retain, y_retain = next(retain_iter)
                x_forget, y_forget = next(forget_iter)
            except StopIteration:
                break

            x_retain, y_retain = x_retain.to(DEVICE), y_retain.to(DEVICE)
            x_forget, y_forget = x_forget.to(DEVICE), y_forget.to(DEVICE)

            optimizer.zero_grad()

            # --- Forward pass retain ---
            out_retain = model(x_retain)
            retain_loss = criterion(out_retain, y_retain)

            # --- Forward pass forget ---
            out_forget = model(x_forget)

            if randomize_labels:
                # Random teacher: mescola o randomizza le etichette
                y_rand = torch.randint_like(y_forget, low=0, high=out_forget.size(1))
                forget_loss = -criterion(out_forget, y_rand)  # loss negativa = dimenticare
            else:
                # Alternativamente, incoraggia uniformità (incertezza)
                probs = F.log_softmax(out_forget, dim=1)
                uniform = torch.full_like(probs, 1.0 / probs.size(1))
                forget_loss = F.kl_div(probs, uniform, reduction='batchmean')

            # --- Combine ---
            loss = alpha * retain_loss + beta * forget_loss

            # --- Backprop ---
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{UNL_EPOCHS}] - Retain Loss: {retain_loss.item():.4f} | Forget Loss: {forget_loss.item():.4f}")

    return model

def unl_one_shot_magnitude(model, forget_loader, retain_loader, prune_ratio=0.2):
    """
    One-Shot Magnitude Prune Unlearning.
    Dimentica selettivamente parti del modello prunando i pesi legati ai dati da dimenticare.
    """

    model.to(DEVICE)
    model.eval()

    # --- Step 1: Calcola gradienti medi sui dati da dimenticare ---
    importance = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            importance[name] = torch.zeros_like(param)

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in forget_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)
            grads = torch.autograd.grad(loss, model.parameters(), retain_graph=False, create_graph=False)
            for (name, param), g in zip(model.named_parameters(), grads):
                if g is not None:
                    importance[name] += g.abs()

    # Normalizza importanza
    for name in importance:
        importance[name] /= len(forget_loader)

    # --- Step 2: Pruning one-shot ---
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    # Applica pruning per ogni modulo in base alla magnitude
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_ratio
    )

    # --- Step 3 (opzionale): Fine-tuning su retain set ---

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(UNL_EPOCHS):
        total_loss = 0
        for x, y in retain_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Fine-tune epoch [{epoch+1}/{UNL_EPOCHS}] - Loss: {total_loss/len(retain_loader):.4f}")

    # --- Step 4: Rimuove i reparametrization buffer per "consolidare" il pruning ---
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

    print(f"✅ One-shot magnitude unlearning completato con prune_ratio={prune_ratio}")
    return model

def unl_amnesiac(model, forget_loader, retain_loader=None, criterion=None, steps=1):
    """
    Amnesiac Unlearning — implementazione PyTorch.
    Effettua aggiornamenti inversi del gradiente per "dimenticare" esempi specifici.
    """

    model.to(DEVICE)
    model.train()

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # --- Step 1: Aggiornamento inverso per i dati da dimenticare ---
    for step in range(steps):
        for x, y in forget_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            model.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)

            # Calcola gradiente
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)

            # Aggiornamento inverso dei pesi (anti-gradient)
            with torch.no_grad():
                for p, g in zip(model.parameters(), grads):
                    if g is not None:
                        p.add_(LR * g)  # direzione inversa rispetto al training normale

        print(f"Unlearning step [{step+1}/{steps}] completato")

    # --- Step 2 (opzionale): Fine-tuning sui dati da mantenere ---

    optimizer = optim.Adam(model.parameters(), lr=LR)
    for epoch in range(UNL_EPOCHS):
        total_loss = 0
        for x, y in retain_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Fine-tune epoch [{epoch+1}/{UNL_EPOCHS}] - Loss: {total_loss/len(retain_loader):.4f}")

    print("Amnesiac unlearning completato.")
    return model

def forget_retain_split(train_ids, train_labels, le):

    idx_to_remove = le.transform([GENRE_TO_FORGET])[0]
    print(f"Rimuovere il genere '{GENRE_TO_FORGET}' (indice {idx_to_remove})")

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

unlearning_main()