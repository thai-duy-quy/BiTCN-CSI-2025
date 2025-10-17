from tqdm import tqdm
import torch

def train(model, dataloader, criterion, optimizer, device, epoch=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"[Epoch {epoch}] Training", leave=False)

    for x, labels in progress_bar:
        #print(x.shape)
        #exit()
        x = x.to(device)           # [B, 4, T, 2025]
        labels = labels.to(device)

        outputs = model(x)         # logits: [B, num_classes]
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy
