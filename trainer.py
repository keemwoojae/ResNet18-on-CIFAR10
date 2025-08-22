import torch
from sklearn.metrics import confusion_matrix

def train(model, optimizer, criterion, train_loader, device):
    model.train()

    train_loss = 0
    correct_preds = 0
    total_samples = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

        preds = torch.argmax(outputs, dim=1)
        total_samples += targets.size(0)
        correct_preds += (preds == targets).sum().item() # preds.eq(targets) 와 같음

    epoch_loss = train_loss / total_samples
    epoch_acc = 100. * correct_preds / total_samples
    return {'loss': epoch_loss,
            'accuracy': epoch_acc}

def evaluate(model, criterion, loader, device, is_test=False):
    model.eval()

    eval_loss = 0
    correct_preds = 0
    total_samples = 0

    # Confusion Matrix 계산을 위한 리스트
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            preds = torch.argmax(outputs, dim=1)

            eval_loss += loss.item() * inputs.size(0)
            total_samples += targets.size(0)
            correct_preds += (preds == targets).sum().item()

            if is_test:
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

    epoch_loss = eval_loss / total_samples
    epoch_acc = 100. * correct_preds / total_samples

    results = {'loss': epoch_loss, 'accuracy': epoch_acc}

    if is_test:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        results['conf_matrix'] = confusion_matrix(all_targets, all_preds)

    return results