# %% [04] Training: evaluation with torchmetrics and training loop
def evaluate_tm(model, data_loader, metric):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    return metric.compute()


def train2(model, optimizer, loss_fn, metric, train_loader, valid_loader, n_epochs):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        model.train()
        metric.reset()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            metric.update(y_pred, y_batch)

        history["train_losses"].append(total_loss / len(train_loader))
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(evaluate_tm(model, valid_loader, metric).item())
        print(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"train loss: {history['train_losses'][-1]:.4f}, "
            f"train accuracy: {history['train_metrics'][-1]:.2%}, "
            f"valid accuracy: {history['valid_metrics'][-1]:.2%}"
        )
    return history


LEARNING_RATE = 0.1
N_EPOCHS = 20

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=len(class_names)).to(device)

history = train2(model, optimizer, criterion, accuracy, train_loader, val_loader, N_EPOCHS)
