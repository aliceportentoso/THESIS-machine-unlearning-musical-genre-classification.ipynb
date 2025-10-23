from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from config import *

def evaluate(model, data_loader, label_encoder):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        print("Testing..")
        for inputs, labels in data_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            outputs = outputs['clipwise_output']
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = compute_accuracy(model, data_loader)
    print(f"Accuracy: {acc:.4f}")

    n_classes = len(label_encoder.classes_)
    cm = confusion_matrix(all_labels, all_preds, labels=range(n_classes))

    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_)
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.xticks(rotation=90)
    if GENRE_TO_REMOVE is None:
        plt.title(f"Confusion Matrix for GENRE CLASSIFICATION, {MAX_EPOCHS} epochs")
    else:
        plt.title(f"Confusion Matrix for LEARNING WITHOUT {GENRE_TO_REMOVE}, {MAX_EPOCHS} epochs")

    plt.show()

    plt.savefig(f"results/CM_{NAME}.png", bbox_inches='tight')  # bbox_inches='tight' evita tagli sulle etichette

    return acc

def evaluate_unlearning(model, forget_loader, retain_loader, val_loader, label_encoder):
    """
    Calcola metriche per l'unlearning:
    - Forget Accuracy
    - Retain Accuracy
    - Global Accuracy
    - Utility Drop
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        print("Testing..")
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            outputs = outputs['clipwise_output']
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n_classes = len(label_encoder.classes_)
    cm = confusion_matrix(all_labels, all_preds, labels=range(n_classes))

    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_)
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.xticks(rotation=90)
    plt.title(f"Confusion Matrix for UNLEARNING of {GENRE_TO_FORGET}, {UNL_EPOCHS} unl epochs")
    plt.show()

    plt.savefig(f"results/CM_{UNL_NAME}.png", bbox_inches='tight')  # bbox_inches='tight' evita tagli sulle etichette

    forget_acc = compute_accuracy(model, forget_loader)
    print(f"Accuracy sui dati da dimenticare: {forget_acc:.4f}") #obiettivo è casuale tipo 1/8
    retain_acc = compute_accuracy(model, retain_loader)
    print(f"Accuracy sui dati rimasti: {retain_acc:.4f}") #obiettivo 35/40 %
    global_acc = compute_accuracy(model, val_loader)
    print(f"Accuracy sui dati totale: {global_acc:.4f}") #obettivo è 40 % non cambia

    #utility_drop = acc_before - global_acc #obiettivo poco tipo 2/3 %
    #print(f"Utility drop: {utility_drop:.4f}")

def compute_accuracy(model, loader):
    """
    Calcola l'accuracy sui dati da dimenticare.
    Più bassa = modello ha dimenticato meglio.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            outputs = outputs['clipwise_output']
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    return acc