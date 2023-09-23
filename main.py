import os

import torch
from torch import nn, optim

from datasets.dataloaders import *
from models.vgg import VGG
from utils import *

dataset_details = {
    "MNIST": [4, 28],
    "CIFAR10": [4, 32],
    "SVHN": [4, 32],
    "TinyImageNet": [20, 64],
}  # {Dataset name: [Total known classes to use, Image size]}
dataset_name = "TinyImageNet"
checkpoint_dir = f"checkpoint/{dataset_name}"

with open("datasets/config.json") as config_file:
    config = json.load(config_file)[dataset_name]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print("Building model...")
net = VGG(
    "VGG19",
    n_classes=dataset_details["TinyImageNet"][0],
    img_size=dataset_details["TinyImageNet"][1],
)

# Move the model to GPU if available
net.to(device)

# Define a loss function and an optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(),
    lr=config["openset_training"]["learning_rate"][1],
    momentum=0.9,
    weight_decay=config["openset_training"]["weight_decay"],
)


# Training
def train(epoch, train_loader, mapping):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        # convert from original dataset label to known class label
        targets = (
            torch.Tensor([mapping[x] for x in targets]).long().to(device)
        )  # Mapping takes only the known classes labels as targets
        optimizer.zero_grad()
        outputs = net(inputs)  # Outputs are the logits here
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(train_loader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (
                train_loss / (batch_idx + 1),
                100.0 * correct / total,
                correct,
                total,
            ),
        )


def val(epoch, val_loader, mapping, trial_num):
    best_acc = 0
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = torch.Tensor([mapping[x] for x in targets]).long().to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(
                batch_idx,
                len(test_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    val_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(state, f"{checkpoint_dir}/{trial_num}_ckpt.pth")
        best_acc = acc


if __name__ == "__main__":
    start_epoch = 0
    max_epoch = config["openset_training"]["max_epoch"][0] + start_epoch
    # max_epoch = 3

    for i in range(5):
        train_loader, val_loader, test_loader, mapping = get_train_loaders(
            dataset_name, trial_num=i, config=config
        )
        print(f"Training {dataset_name} dataset for {i}th trial")
        for epoch in range(start_epoch, max_epoch):
            train(epoch, train_loader=train_loader, mapping=mapping)
            val(epoch, val_loader=val_loader, mapping=mapping, trial_num=i)
