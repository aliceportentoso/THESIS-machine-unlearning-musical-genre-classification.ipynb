import joblib
from torch.utils.data import DataLoader

from eval import evaluate_unlearning
from eval import evaluate
from dataset import FMADataset
from model import Cnn6
from config import *

def unlearning_main(label_encoder):
    # --- CARICA MODELLO E LABEL ENCODER ---
    model = Cnn6().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
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

#    forget_ids = [148]  # esempi di track_ids da dimenticare
#    forget_labels = [1]  # etichette corrispondenti

    # questo per fare dimenticare genere
    genre_names = ["Electronic", "Experimental", "Folk", "Hip-Hop",
                   "Instrumental", "International", "Pop", "Rock"]
    GENRE_TO_FORGET = "Hip-Hop"  # <-- cambia qui il genere
    genre_index = genre_names.index(GENRE_TO_FORGET)

    # prendi 100 indici casuali (senza ripetizioni)
    #random_indices = np.random.choice(len(train_ids), size=NUM_FORGET, replace=False)

    # ottieni gli ID e le labels corrispondenti
    #forget_ids = [train_ids[i] for i in random_indices]
    #forget_labels = [train_labels[i] for i in random_indices]
    #retain_ids = [x for x in train_ids if x not in forget_ids]
    #retain_labels = [y for (x, y) in zip(train_ids, train_labels) if x not in forget_labels]

    # QUESTO PER I GENERI
    forget_ids = [tid for tid, label in zip(train_ids, train_labels) if label == genre_index]
    forget_labels = [label for label in train_labels if label == genre_index]
    retain_ids = [tid for tid, label in zip(train_ids, train_labels) if label != genre_index]
    retain_labels = [label for label in train_labels if label != genre_index]

    retain_dataset = FMADataset(retain_ids, retain_labels)
    forget_dataset = FMADataset(forget_ids, forget_labels)
    val_dataset = FMADataset(val_ids, val_labels)

    retain_loader = DataLoader(retain_dataset, batch_size=BATCH_SIZE, shuffle=False)
    forget_loader = DataLoader(forget_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- OTTIMIZZATORE E LOSS ---
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- applice algoritmo di UNLEARNING ---
    #unl_fine_tuning(model, forget_loader, criterion, optimizer)
    unl_gradient_ascent(model, forget_loader, retain_loader, criterion, optimizer)

    # --- evaluate ---
    metrics = evaluate_unlearning(model, forget_loader, retain_loader, val_loader, accuracy_train)
    print(f"num forget {NUM_FORGET}")

    # --- SALVA MODELLO AGGIORNATO ---
    torch.save(model.state_dict(), UNL_MODEL_PATH)
    evaluate(model, forget_loader, label_encoder)
    print(f"Modello aggiornato salvato in {UNL_MODEL_PATH}")

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

def unl_gradient_ascent(model, forget_loader, retain_loader, criterion, optimizer, alpha=0.5, beta=0.5):
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

#unlearning_main()