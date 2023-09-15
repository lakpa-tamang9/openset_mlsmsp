import json

import numpy as np
import torch

from datasets.dataloaders import get_eval_loaders
from models.vgg import VGG
from utils import *

start_trial = 0
num_trials = 1
dataset_name = "CIFAR10"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_scores(
    net,
    mapping,
    dataloader,
    unknown=False,
    use_mls=False,
    use_stmls=False,
):
    """Tests data and returns outputs and their ground truth labels.
    unknown : True if an unknown dataset
    """
    X = []
    y = []

    for i, data in enumerate(dataloader):
        images, labels = data
        images = images.to(device)

        if unknown:
            targets = labels
        else:
            targets = torch.Tensor([mapping[x] for x in labels]).long().to(device)

        outputs = net(images)
        logits = outputs[0]
        _, predicted = torch.max(logits, 1)
        mask = predicted == targets
        logits = logits[mask]
        targets = targets[mask]

        if use_mls:
            scores = get_max_logits(logits=logits)
        elif use_stmls:
            scores = get_standardized_max_logits(logits=logits)

        scores = get_softmax_prob(logits=logits)

        X += scores.cpu().detach().tolist()
        y += targets.cpu().tolist()

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y


for trial in range(start_trial, start_trial + num_trials):
    with open("./datasets/config.json") as f:
        config = json.load(f)[dataset_name]

    known_loader, unknown_loader, mapping = get_eval_loaders(
        dataset_name, trial, config
    )
    net = VGG("VGG19", n_classes=4)
    checkpoint = torch.load(f"checkpoints/{dataset_name}/{trial}_ckpt.pth")

    net = net.to(device)
    print("test")

    net_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint["net"].items() if k in net_dict}

    net.load_state_dict(pretrained_dict)
    net.eval()

    known_pred_scores, known_pred_labels = predict_scores(net, mapping, known_loader)
    print(f"Evaluating closed set accuracy for trial {trial}")
    accuracy = get_accuracy(known_pred_scores, known_pred_labels)

    unknown_pred_scores, unknown_pred_labels = predict_scores(
        net, mapping, unknown_loader
    )
    print(f"Evaluating open set AUROC for trial {trial}")
    auroc = get_auroc(known_pred_scores, unknown_pred_scores)
