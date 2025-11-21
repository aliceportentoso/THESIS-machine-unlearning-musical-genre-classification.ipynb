import joblib
from eval import evaluate_unlearning
from dataset import FMADataset
from model import Cnn6
import time
from train import *
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from config import *
import torch.optim as optim
import torch.nn.functional as F

def unlearning_main(prune_ratio, ft_epochs):
    retain_ids, forget_ids, retain_labels, forget_labels = [], [], [], []
    start_time = time.time()

    # --- CARICA MODELLO E LABEL ENCODER ---
    model = Cnn6().to(DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=DEVICE)) # carica i pesi salvati dall'addestramento

    model.eval()
    le = joblib.load(Config.ENCODER_PATH)

    # carica gli split
    dir_ = f"data_splits/{Config.SUBSET}-dataset_remove-None"
    train_ids = joblib.load(f"{dir_}/train_ids.joblib")
    train_labels = joblib.load(f"{dir_}/train_labels.joblib")
    val_ids = joblib.load(f"{dir_}/val_ids.joblib")
    val_labels = joblib.load(f"{dir_}/val_labels.joblib")

    if Config.GENRE_TO_FORGET is not None: # dividi il train in cosa tenere e cosa dimenticare
        forget_ids, forget_labels, retain_ids, retain_labels = forget_retain_split(train_ids, train_labels, le)

    retain_dataset = FMADataset(retain_ids, retain_labels)
    forget_dataset = FMADataset(forget_ids, forget_labels)
    val_dataset = FMADataset(val_ids, val_labels)

    retain_loader = DataLoader(retain_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    forget_loader = DataLoader(forget_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)

    # --- ALGORITMI DI UNLEARNING ---
    if Config.UNL_METHOD == "FT":
        unl_fine_tuning(model, forget_loader, retain_loader, val_loader, criterion, optimizer, le)
    elif Config.UNL_METHOD == "GA":
        unl_gradient_ascent(model, forget_loader, retain_loader, val_loader, criterion, optimizer, le)
    elif Config.UNL_METHOD == "ST":
        unl_stochastic_teacher(model, forget_loader, retain_loader, val_loader, criterion, optimizer, le)
    elif Config.UNL_METHOD == "OSM":
        unl_one_shot_magnitude(model, forget_loader, retain_loader, val_loader, criterion, optimizer, le, prune_ratio, ft_epochs)
    elif Config.UNL_METHOD == "A":
        unl_amnesiac(model, forget_loader, retain_loader, criterion=None, steps=1)
    else:
        print("unknown method")

    print(f"Tempo Unlearning: {(time.time() - start_time)/3600:.2f} ore")

def unl_fine_tuning(model, forget_loader, retain_loader, val_loader, criterion, optimizer, le, lambda_unlearn=0.5):
    model.train()
    forget_losses, retain_losses = [], []
    forget_accs, retain_accs = [], []
    retain_iter = iter(retain_loader)

    for epoch in range(Config.UNL_EPOCHS):
        total_forget, total_retain = 0.0, 0.0
        num_batches = 0

        for forget_data in forget_loader:
            try:
                retain_data = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_data = next(retain_iter)

            # === Forget batch ===
            x_f, y_f = [d.to(DEVICE) for d in forget_data]
            out_f = model(x_f)
            if isinstance(out_f, dict) and "clipwise_output" in out_f:
                out_f = out_f["clipwise_output"]
            loss_f = criterion(out_f, y_f)

            # === Retain batch ===
            x_r, y_r = [d.to(DEVICE) for d in retain_data]
            out_r = model(x_r)
            if isinstance(out_r, dict) and "clipwise_output" in out_r:
                out_r = out_r["clipwise_output"]
            loss_r = criterion(out_r, y_r)

            # === Combined loss ===
            loss = loss_r - lambda_unlearn * loss_f

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_forget += loss_f.item()
            total_retain += loss_r.item()
            num_batches += 1

        avg_f = total_forget / num_batches
        avg_r = total_retain / num_batches
        forget_losses.append(avg_f)
        retain_losses.append(avg_r)

        print(f"Epoch {epoch+1}/{Config.UNL_EPOCHS} | Retain: {avg_r:.4f} | Forget: {avg_f:.4f}")
    f_acc, r_acc = evaluate_unlearning(model, forget_loader, retain_loader, val_loader, le)
        #forget_accs.append(f_acc)
        #retain_accs.append(r_acc)
        #print_loss(forget_losses, retain_losses, forget_accs, retain_accs, unlearning=True)

    print("FINE TUNING completato.")
    return forget_losses, retain_losses


def unl_gradient_ascent(model, forget_loader, retain_loader, val_loader, criterion, optimizer, le, alpha=0.4, beta=0.6):
    """
    - Usa gradient ascent sui dati da dimenticare. Usa gradient descent sui dati da mantenere (regolarizzazione).
    - Controlla il bilanciamento tramite i pesi alpha e beta.    """
    model.train()
    forget_losses, retain_losses = [], []
    forget_accs, retain_accs = [], []
    total_forget, total_retain = 0.0, 0.0
    num_batches = 0

    for epoch in range(Config.UNL_EPOCHS):
        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)
        epoch_loss = 0.0
        total_batches = 0

        for _ in range(min(len(forget_loader), len(retain_loader))):
            # batch da dimenticare
            try:
                f_inputs, f_labels = next(forget_iter)
                f_inputs, f_labels = f_inputs.to(DEVICE), f_labels.to(DEVICE)
            except StopIteration:
                break

            # batch da mantenere
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

            total_forget += f_loss.item()
            total_retain += r_loss.item()
            num_batches += 1

            loss = -alpha * f_loss + beta * r_loss
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        total_batches += 1

        avg_f = total_forget / num_batches
        avg_r = total_retain / num_batches
        forget_losses.append(avg_f)
        retain_losses.append(avg_r)

        print(f"Epoch {epoch + 1}/{Config.UNL_EPOCHS} | Retain: {avg_r:.4f} | Forget: {avg_f:.4f}")
        f_acc, r_acc = evaluate_unlearning(model, forget_loader, retain_loader, val_loader, forget_losses, retain_losses, le)
        forget_accs.append(f_acc)
        retain_accs.append(r_acc)
        print_loss(forget_losses, retain_losses, forget_accs, retain_accs, unlearning=True)

    print("GRADIENT ASCENT completato.")
    return forget_losses, retain_losses

def unl_stochastic_teacher(model, forget_loader, retain_loader, val_loader, criterion, optimizer, le, alpha=0.3, beta=0.7, randomize_labels=False):
    """   randomize_labels: se True, randomizza le label dei dati da dimenticare  """
    print(f'alpha: {alpha}')
    print(f'beta: {beta}')
    model.train()
    model.to(DEVICE)

    forget_losses, retain_losses = [], []
    forget_accs, retain_accs = [], []

    for epoch in range(Config.UNL_EPOCHS):
        retain_iter = iter(retain_loader)
        forget_iter = iter(forget_loader)

        num_batches = min(len(retain_iter), len(forget_iter))

        for _ in range(Config.BATCH_SIZE):
            try:
                x_retain, y_retain = next(retain_iter)
                x_forget, y_forget = next(forget_iter)
            except StopIteration:
                break

            x_retain, y_retain = x_retain.to(DEVICE), y_retain.to(DEVICE)
            x_forget, y_forget = x_forget.to(DEVICE), y_forget.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass retain e forget
            out_retain = model(x_retain)['clipwise_output']
            retain_loss = criterion(out_retain, y_retain)
            out_forget = model(x_forget)['clipwise_output']

            if randomize_labels:
                # Random teacher: mescola o randomizza le etichette
                y_rand = torch.randint_like(y_forget, low=0, high=out_forget.size(1))
                forget_loss = -criterion(out_forget, y_rand)  # loss negativa = dimenticare
            else:
                # Alternativamente, incoraggia uniformità (incertezza)
                probs = F.log_softmax(out_forget, dim=1)
                uniform = torch.full_like(probs, 1.0 / probs.size(1))
                forget_loss = F.kl_div(probs, uniform, reduction='batchmean')

            loss = alpha * retain_loss + beta * forget_loss
            loss.backward()
            optimizer.step()

        forget_losses.append(forget_loss)
        retain_losses.append(retain_loss)

        print(f"Epoch {epoch + 1}/{Config.UNL_EPOCHS} | Retain: {retain_loss:.4f} | Forget: {forget_loss:.4f}")
        f_acc, r_acc = evaluate_unlearning(model, forget_loader, retain_loader, val_loader, forget_losses, retain_losses, le)
        forget_accs.append(f_acc)
        retain_accs.append(r_acc)
        print_loss(forget_losses, retain_losses, forget_accs, retain_accs, unlearning=True)

    print("STOCHASTICH TEACHER completato.")
    return forget_losses, retain_losses

def unl_one_shot_magnitude(model, forget_loader, retain_loader, val_loader, criterion, optimizer, le, prune_ratio, ft_epochs):

    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))
            if hasattr(module, 'bias') and module.bias is not None:
                parameters_to_prune.append((module, 'bias'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_ratio
    )

    # --- Step 2: Bloccare i pesi azzerati con una maschera
    for module, param_name in parameters_to_prune:
        mask = getattr(module, f"{param_name}_mask")
        module.register_buffer(f"{param_name}_mask_blocked", mask.clone())
        # Override backward per blocco (semplice esempio)
        param = getattr(module, param_name)
        param.grad = None
        param.register_hook(lambda grad, mask=mask: grad * mask)

    # --- Step 3: Fine-tuning leggero sui dati da mantenere
    model.train()
    # fine-tuning rapido
    for epoch in range(ft_epochs):
        for x, y in retain_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)['clipwise_output']
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    evaluate_unlearning(model, forget_loader, retain_loader, val_loader, le)

def unl_amnesiac(model, forget_loader, retain_loader=None, criterion=None, steps=1):
    """   Effettua aggiornamenti inversi del gradiente per "dimenticare" esempi specifici.  """

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
            outputs = outputs['clipwise_output']
            loss = criterion(outputs, y)

            for name, param in model.named_parameters():
                print(param.requires_grad)

            # Calcola gradiente
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)

            # Aggiornamento inverso dei pesi (anti-gradient)
            with torch.no_grad():
                for p, g in zip(model.parameters(), grads):
                    if g is not None:
                        p.add_(Config.LR * g)  # direzione inversa rispetto al training normale

        print(f"Unlearning step [{step+1}/{steps}] completato")

    # --- Step 2 (opzionale): Fine-tuning sui dati da mantenere ---

    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    for epoch in range(Config.UNL_EPOCHS):
        total_loss = 0
        for x, y in retain_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Fine-tune epoch [{epoch+1}/{Config.UNL_EPOCHS}] - Loss: {total_loss/len(retain_loader):.4f}")

    print(f"Complete UNLEARNING con AMNESIAC")
    return model

def forget_retain_split(train_ids, train_labels, le):

    idx_to_remove = le.transform([Config.GENRE_TO_FORGET])[0]
    print(f"Rimuovere il genere '{Config.GENRE_TO_FORGET}' (indice {idx_to_remove})")

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

if Config.UNL_METHOD == "FT":
    for genre in Config.GENRES:
        Config.GENRE_TO_FORGET = genre
        Config.unl_name_path()
        Config.print_config_unl()
        unlearning_main(1,1)

if Config.UNL_METHOD == "OSM":
    prunes = [0.1, 0.3, 0.5, 0.7, 0.9]
    FT_epochs = [0,1,2,3]
    for ft_epochs in FT_epochs:
        for prune_ratio in prunes :
            Config.UNL_NAME = f"OSM/FT_epochs_{ft_epochs}-prune_ratio_{prune_ratio}"
            print(f"FT epochs: {ft_epochs} prune ratio: {prune_ratio}")
            Config.print_config_unl()
            unlearning_main(prune_ratio, ft_epochs)
