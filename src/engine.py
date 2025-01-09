from fastprogress.fastprogress import master_bar, progress_bar
import torch


def train_model(
    model, train_dl, valid_dl, loss_fn, num_epochs, optimizer, scheduler, device
):
    mb = master_bar(range(num_epochs))
    for epoch in mb:
        model.train()
        running_loss = 0
        running_acc = 0
        val_run_loss = 0
        val_run_acc = 0
        for xb, yb in progress_bar(train_dl, parent=mb):
            logits = model(xb.to(device))
            loss = loss_fn(logits, torch.argmax(yb.to(device), dim=1))
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predictions == yb).all(dim=1).float().mean()
            running_acc += accuracy.item()
        scheduler.step()

        for xb, yb in progress_bar(valid_dl, parent=mb):
            with torch.no_grad():
                logits = model(xb.to(device))
                loss = loss_fn(logits, torch.argmax(yb.to(device), dim=1))
                val_run_loss += loss.item()

                predictions = (torch.sigmoid(logits) > 0.5).float()
                accuracy = (predictions == yb).all(dim=1).float().mean()
                val_run_acc += accuracy.item()
        mb.write(
            f"Epoch {epoch} | "
            f"Train Loss: {running_loss/len(train_dl):.3f} | "
            f"Valid Loss: {val_run_loss/len(valid_dl):.3f} | "
            f"Train Acc: {running_acc/len(train_dl):.2f} | "
            f"Valid Acc: {val_run_acc/len(valid_dl):.2f}"
        )
