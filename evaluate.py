import json

import numpy as np
import torch

torch.cuda.empty_cache()

from datasets.dataloaders import get_eval_loaders
from models.vgg import VGG
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def test(trial_num, dataset_name, use_mls=False, use_stmls=False):
    all_accuracy = []
    with open("./datasets/config.json") as f:
        config = json.load(f)[dataset_name]

    known_loader, unknown_loader, mapping = get_eval_loaders(
        dataset_name, trial_num, config
    )
    net = VGG("VGG19", n_classes=4)
    checkpoint = torch.load(
        f"checkpoint/{dataset_name}/{trial_num}_ckpt.pth",
        map_location=torch.device(device),
    )
    # map_location = torch.device("cpu")

    net = net.to(device)
    net_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint["net"].items() if k in net_dict}

    net.load_state_dict(pretrained_dict)
    net.eval()

    X = []
    y = []
    known_max_logits = []
    known_std_max_logits = []

    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for _, data in enumerate(known_loader):
            images, labels = data
            targets = torch.Tensor([mapping[x] for x in labels]).long().to(device)

            images = images.to(device)
            logits = net(images)

            if use_mls:
                known_max_logit, _ = torch.max(logits, dim=1)
                pred_labels = torch.argmax(logits, dim=1)
            elif use_stmls:
                logits = get_standardized_max_logits(logits=logits)
                pred_labels = torch.argmax(logits)
                known_std_max_logit, _ = torch.max(logits, dim=1)
            else:
                pred_labels = softmax(logits)

            if use_mls or use_stmls:
                X += logits.cpu().detach().tolist()
            else:
                X += pred_labels.cpu().detach().tolist()

            y += targets.cpu().tolist()

            if use_mls:
                known_max_logits += known_max_logit
            elif use_stmls:
                known_std_max_logits += known_std_max_logit

        # Extract items from the list of tensors
        if use_mls:
            known_max_logits = [t.item() for t in known_max_logits]
        elif use_stmls:
            known_std_max_logits = [t.item() for t in known_std_max_logits]

        X = np.asarray(X)
        y = np.asarray(y)

        accuracy = get_accuracy(X, y, use_mls, use_stmls)
        all_accuracy += [accuracy]

    Xu = []
    unknown_max_logits = []
    with torch.no_grad():
        for _, data in enumerate(unknown_loader):
            images, labels = data

            images = images.to(device)
            logits = net(images)
            if use_mls:
                unknown_max_logit, _ = torch.max(logits, dim=1)
                pred_labels = torch.argmax(logits, dim=1)

            elif use_stmls:
                logits = get_standardized_max_logits(logits=logits)
                pred_labels = torch.argmax(logits)
                unknown_std_max_logit, _ = torch.max(logits, dim=1)

            else:
                pred_labels = softmax(logits)

            if use_mls:
                unknown_max_logits += unknown_max_logit
            elif use_stmls:
                unknown_std_max_logits += unknown_std_max_logit

            Xu += pred_labels.cpu().detach().tolist()

        if use_mls:
            unknown_max_logits = [t.item() for t in unknown_max_logits]
        elif use_stmls:
            unknown_std_max_logits = [t.item() for t in unknown_std_max_logits]

        Xu = np.asarray(Xu)

        if use_mls:
            known_class_logits = known_max_logits
            unknown_class_logits = unknown_max_logits

        elif use_stmls:
            known_class_logits = known_std_max_logits
            unknown_class_logits = unknown_std_max_logits

        else:
            known_class_logits = X
            unknown_class_logits = Xu

        auc = calculate_auroc(known_class_logits, unknown_class_logits)

    mean_acc = np.mean(all_accuracy)
    mean_auc = np.mean(auc)

    return mean_acc, mean_auc


if __name__ == "__main__":
    dataset_name = "CIFAR10"
    eval_dict = {}
    trials = []
    for i in range(5):
        print("Testing for Trial {}".format(i))
        accuracy, auc = test(
            trial_num=i,
            dataset_name=dataset_name,
            use_mls=True,
        )
        trials.append([accuracy, auc])
    eval_dict[f"{dataset_name}"] = trials
    print(eval_dict)
    with open("./eval_dict.json", "w") as f:
        json.dump(eval_dict, f)
