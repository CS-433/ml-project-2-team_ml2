from src.Save_Load.load_data import *
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.Modeles.ResUNet import *
# Load data
def get_dataloaders(split,frac_data):
    data_dataset = Roads(split=split, frac_data=frac_data)
    #print(f"Data_set size : {len(data_dataset)}")
    # data loader
    data_loader = DataLoader(data_dataset,
                             batch_size=3,
                             shuffle=True,
                             num_workers=0,
                             pin_memory=False)
    return data_dataset, data_loader
# Initialize neural network

def train_epoch(model, optimizer, scheduler, criterion, train_loader, epoch, device):
    model.train()
    loss_history = []
    accuracy_history = []
    lr_history = []
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.type(torch.LongTensor)  # avoid an error idk why?
        # data = data.float()
        data, target = data.float().to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        '''print("Original shape of output and target:")
        print(output.shape)
        print(target.shape)'''
        output = output.flatten().float()  # [batch*400*400]
        target = target.flatten().float()  # [batch*400*400]
        '''print(f"Output: min = {torch.min(output)}; max = {torch.max(output)}")
        print(f"Target: min = {torch.min(target)}; max = {torch.max(target)}")
        print(torch.mean(output))
        print(torch.mean(target))
        print("Shape after flatten of output and target:")
        print(output.shape)
        print(target.shape)'''

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # predictions = output.argmax(1).cpu().detach().numpy()
        predictions = output.cpu().detach().numpy()
        ground_truth = target.cpu().detach().numpy()

        accuracy_float = (predictions == ground_truth).mean()
        loss_float = loss.item()

        loss_history.append(loss_float)
        accuracy_history.append(accuracy_float)
        lr_history.append(scheduler.get_last_lr()[0])
        if batch_idx % (len(train_loader.dataset) // len(data) // 10) == 0:
            print(
                f"Train Epoch: {epoch}-{batch_idx:03d} "
                f"batch_loss={loss_float:0.2e} "
                f"batch_acc={accuracy_float:0.3f} "
                f"lr={scheduler.get_last_lr()[0]:0.3e} "
            )

    return loss_history, accuracy_history, lr_history


@torch.no_grad()
def validate(model, device, val_loader, criterion):
    model.eval()  # Important: eval mode (affects dropout, batch norm etc)
    test_loss = 0
    accuracy_float = 0
    for data, target in val_loader:
        target = target.type(torch.LongTensor)  # avoid an error idk why?
        data = data.float()
        data, target = data.to(device), target.to(device)
        output = model(data)
        output = output.flatten().float()  # [batch,class,200*200]
        target = target.flatten().float()  # [batch,200*200] expected for criterion argument (class per pixel)
        """print(output.shape)
        print(target.shape)"""
        test_loss += criterion(output, target)

        # predictions = output.argmax(1).cpu().detach().numpy()
        predictions = output.cpu().detach().numpy()
        ground_truth = target.cpu().detach().numpy()

        accuracy_float += (predictions == ground_truth).mean()

    test_loss /= len(val_loader)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            accuracy_float,
            len(val_loader),
            100.0 * accuracy_float / len(val_loader),
        )
    )
    return test_loss, accuracy_float / len(val_loader)


def run_training(model_factory, num_epochs, optimizer_kwargs, device="cuda", frac_data='1.0'):
    # ===== Data Loading =====
    train_loader = get_dataloaders("train", frac_data)[1]
    val_loader = get_dataloaders("val", frac_data)[1]
    print(f"Data_set size : {len(train_loader.dataset)}")
    print(f"Data_set size : {len(val_loader.dataset)}")

    # ===== Model, Optimizer and Criterion =====
    model = resunet(3, 1)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    criterion = torch.nn.functional.cross_entropy
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(len(train_loader.dataset) * num_epochs) // train_loader.batch_size,
    )

    # ===== Train Model =====
    lr_history = []
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, lrs = train_epoch(
            model, optimizer, scheduler, criterion, train_loader, epoch, device
        )
        train_loss_history.extend(train_loss)
        train_acc_history.extend(train_acc)
        lr_history.extend(lrs)

        val_loss, val_acc = validate(model, device, val_loader, criterion)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

    return sum(train_acc) / len(train_acc), val_acc_history, model

