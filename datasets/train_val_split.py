import json
import random
import torchvision
import numpy as np
import os

random.seed(1000)


def save_trainval_split(dataset, train_idxs, val_idxs):
    global test_idxs
    print("Saving {} Train/Val split to {}/trainval_idxs.json".format(dataset, dataset))
    CWD = os.getcwd()
    if dataset == "TinyImageNet":
        split_data = {"Train": train_idxs, "Val": val_idxs, "Test": test_idxs}
    else:
        split_data = {"Train": train_idxs, "Val": val_idxs}
    with open(f"{CWD}/datasets/{dataset}/trainval_idxs.json", "w") as f:
        json.dump(split_data, f)


mnist = torchvision.datasets.MNIST("datasets/data", download=True)
# svhn = torchvision.datasets.SVHN("data")
cifar10 = torchvision.datasets.CIFAR10("datasets/data", download=False)
tinyImagenet = torchvision.datasets.ImageFolder("datasets/data/tiny-imagenet-200/train")

datasets = {
    # "MNIST": mnist,
    # "SVHN": svhn,
    # "CIFAR10": cifar10,
    "TinyImageNet": tinyImagenet,
}
split = {"MNIST": 0.8, "SVHN": 0.8, "CIFAR10": 0.8, "TinyImageNet": 0.7}
# split = {
#     "MNIST": 0.8,
#     "CIFAR10": 0.8,
# }

for datasetName in datasets.keys():
    dataset = datasets[datasetName]

    # get class label for each image. svhn has different syntax as .labels
    try:
        targets = dataset.targets
        num_classes = len(dataset.classes)
    except:
        targets = dataset.labels
        num_classes = len(np.unique(targets))

    # save image idxs per class
    class_idxs = [[] for i in range(num_classes)]
    for i, lbl in enumerate(targets):
        class_idxs[lbl] += [i]

    # determine size of train subset
    class_size = [len(x) for x in class_idxs]
    class_train_size = [int(split[datasetName] * x) for x in class_size]
    if (
        datasetName == "TinyImageNet"
    ):  # Separate 0.7 for train, 0.2 for val and 0.1 for test from the Train directory itself.
        class_val_size = [int(0.2 * x) for x in class_size]
        class_test_size = [int(0.1 * x) for x in class_size]

    # subset per class into train and val subsets randomly
    train_idxs = {}
    val_idxs = {}
    test_idxs = {}
    for class_num in range(num_classes):
        train_size = class_train_size[class_num]
        idxs = class_idxs[class_num]
        random.shuffle(idxs)
        if datasetName == "TinyImageNet":
            val_size = class_val_size[class_num]
            test_size = class_test_size[class_num]

            train_idxs[class_num] = idxs[:train_size]
            val_idxs[class_num] = idxs[train_size : train_size + val_size]
            test_idxs[class_num] = idxs[
                train_size + val_size : train_size + val_size + test_size
            ]

        train_idxs[class_num] = idxs[:train_size]
        val_idxs[class_num] = idxs[train_size:]

    save_trainval_split(datasetName, train_idxs, val_idxs)

    # cifar10 and cifar+m datasets can use the same training and val splits
    # if "CIFAR" in datasetName:
    #     save_trainval_split("CIFAR+10", train_idxs, val_idxs)
    #     save_trainval_split("CIFAR+50", train_idxs, val_idxs)
