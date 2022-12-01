#!/usr/bin/python


# Import libraries
# TODO Importer depuis src... @Anthony J'ai pas trouvé comment faire, du coup j'ai déplacé les fichiers .py dans root
from load_data import *
from torch.utils.data import DataLoader
import sys
sys.path.append('./src/Modeles')
sys
from UNet import *
# Load data
print("Loading data...")
input_dir_obs = "Data/training_processed/images"
input_dir_label = "Data/training_processed/groundtruth"
obs = load_data(input_dir_obs, img_size=400)
label = load_data(input_dir_label, img_size=400)


def get_dataloaders(split):
    data_dataset = Roads()
    print(f"Data_set size : {len(data_dataset)}")
    # data loader
    data_loader = DataLoader(data_dataset,
                             batch_size=4,
                             shuffle=True,
                             num_workers=0,
                             pin_memory=False)
    return data_dataset, data_loader
# Label to binary, unique channel
label = label[:, 1] > 0
label = label[:, None, :, :]  # Ajoute la dim C = 1
print("Data Loaded!")

#Check for gpu availability:
if torch.cuda.is_available():
    print("CUDA IS AVAILABLE!")
else:
    print("WARNING: CUDA NOT AVAILABLE!")

# Initialize neural network

def train_epoch(model, optimizer, scheduler, criterion, train_loader, epoch, device):
    model.train()
    loss_history = []
    accuracy_history = []
    lr_history = []
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.type(torch.LongTensor)  # avoid an error idk why?
        data = data.float()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        '''print("Original shape of output and target:")
        print(output.shape)
        print(target.shape)'''
        output = output.flatten(2).float()  #[batch,class*400*400]
        target = target.flatten(2).float()  #[batch,class*400*400]
        '''print("Shape after flatten of output and target:")
        print(output.shape)
        print(target.shape)'''

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        #predictions = output.argmax(1).cpu().detach().numpy()
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
        output = output.flatten(2)  # [batch,class,200*200]
        target = target.flatten(-3)  # [batch,200*200] expected for criterion argument (class per pixel)

        test_loss += criterion(output, target)

        predictions = output.argmax(1).cpu().detach().numpy()
        ground_truth = target.cpu().detach().numpy()

        accuracy_float += (predictions == ground_truth).mean()

    test_loss /= len(val_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            accuracy_float,
            len(val_loader.dataset),
            100.0 * accuracy_float / len(val_loader.dataset),
        )
    )
    return test_loss, accuracy_float / len(val_loader.dataset)


def run_training(model_factory, num_epochs, optimizer_kwargs, device="cuda"):
    # ===== Data Loading =====
    train_loader = get_dataloaders("train")[1]
    val_loader = get_dataloaders("val")[1]

    # ===== Model, Optimizer and Criterion =====
    model = UNet(3, 3)
    model = model.to(device=device)
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

# U_net

# Train neural network
torch.cuda.empty_cache()
image_size = 400
model_factory = UNet
num_epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer_kwargs = dict(
    lr=1e-3,
    weight_decay=1e-2,
)

train_acc,val_acc,model = run_training(
    model_factory=UNet,
    num_epochs=num_epochs,
    optimizer_kwargs=optimizer_kwargs,
    device=device,
)
# Make-save prediction
