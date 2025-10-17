from tqdm import tqdm
import torch

def evaluate(model, dataloader, criterion, device, epoch=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc=f"[Epoch {epoch}] Evaluating", leave=False)

    with torch.no_grad():
        for x, labels in progress_bar:
            x = x.to(device)           # [B, 4, T, 2025]
            labels = labels.to(device)

            outputs = model(x)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)
            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy, all_preds, all_labels
