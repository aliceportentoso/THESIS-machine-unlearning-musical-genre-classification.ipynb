from tqdm import tqdm
import matplotlib.pyplot as plt
from config import *

def train(model, train_loader, val_loader, criterion, optimizer, device):
    patience = 30
    best_val_loss = float("inf")
    patience_counter = 0

    # liste per i plot
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(MAX_EPOCHS):
        # ---- TRAIN ----
        model.train()
        running_loss, correct, total = 0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{MAX_EPOCHS}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs['clipwise_output']
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ---- VALIDATION ----
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs = outputs['clipwise_output']
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch + 1}/{MAX_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        best_weights = 0
        # ---- EARLY STOPPING ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = model.state_dict().copy()
        else:
            patience_counter += 1
            print(f"Patience_counter: {patience_counter}")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                model.load_state_dict(best_weights)
                break

        # ---- PLOT IN TEMPO REALE ----
        if epoch % 5 == 0:
            plt.figure(figsize=(12, 5))

            # Loss
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Val Loss")
            plt.legend()
            plt.title("Loss")

            # Accuracy
            plt.subplot(1, 2, 2)
            plt.plot(train_accs, label="Train Acc")
            plt.plot(val_accs, label="Val Acc")
            plt.legend()
            plt.title("Accuracy")

            plt.show()


    return model, (train_losses, val_losses, train_accs, val_accs)