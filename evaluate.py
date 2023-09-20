import json

import numpy as np
import torch

torch.cuda.empty_cache()

from datasets.dataloaders import get_eval_loaders
from models.vgg import VGG
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def test(trial_num, num_trials, dataset_name, use_mls=False, use_stmls=False):
    all_accuracy = []
    all_auroc = []
    for trial in range(trial_num, trial_num + num_trials):
        with open("./datasets/config.json") as f:
            config = json.load(f)[dataset_name]

        known_loader, unknown_loader, mapping = get_eval_loaders(
            dataset_name, trial, config
        )
        net = VGG("VGG19", n_classes=4)
        checkpoint = torch.load(
            f"checkpoint/{dataset_name}/trial_{trial}_ckpt.pth",
            map_location=torch.device(device),
        )
        # map_location = torch.device("cpu")

        net = net.to(device)
        print("test")

        net_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint["net"].items() if k in net_dict}

        net.load_state_dict(pretrained_dict)
        net.eval()

        X = []
        X_roc = []
        y = []

        softmax = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            for i, data in enumerate(known_loader):
                images, labels = data
                targets = torch.Tensor([mapping[x] for x in labels]).long().to(device)

                images = images.to(device)
                logits = net(images)
                if use_mls:
                    scores = torch.argmax(logits, dim=1)
                elif use_stmls:
                    std_logits = get_standardized_max_logits(logits=logits)
                    scores = torch.argmax(std_logits)
                else:
                    scores = softmax(logits)

                if use_mls or use_stmls:
                    X += logits.cpu().detach().tolist()
                else:
                    X += scores.cpu().detach().tolist()
                y += targets.cpu().tolist()

            X = np.asarray(X)
            y = np.asarray(y)

            accuracy = get_accuracy(X, y, use_mls, use_stmls)
            all_accuracy += [accuracy]

        Xu = []
        with torch.no_grad():
            for i, data in enumerate(unknown_loader):
                images, labels = data

                images = images.to(device)
                logits = net(images)
                if use_mls:
                    scores = torch.argmax(logits, dim=1)
                elif use_stmls:
                    std_logits = get_standardized_max_logits(logits=logits)
                    scores = torch.argmax(std_logits)
                else:
                    scores = softmax(logits)

                if use_mls or use_stmls:  # Use logits for mls or stmls
                    Xu += logits.cpu().detach().tolist()
                else:
                    Xu += scores.cpu().detach().tolist()

            Xu = np.asarray(Xu)

            auroc = get_auroc(X, Xu)
            all_auroc += [auroc]

    mean_acc = np.mean(all_accuracy)
    mean_auroc = np.mean(all_auroc)

    print("Raw Top-1 Accuracy: {}".format(all_accuracy))
    print("Raw AUROC: {}".format(all_auroc))
    print("Average Top-1 Accuracy: {}".format(mean_acc))
    print("Average AUROC: {}".format(mean_auroc))


if __name__ == "__main__":
    # trial_num = 0
    num_trials = 1
    dataset_name = "CIFAR10"

    # for i in range(5):
    test(trial_num=0, num_trials=num_trials, dataset_name=dataset_name, use_stmls=True)
