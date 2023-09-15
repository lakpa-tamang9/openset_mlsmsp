import torch
import torchvision
from torchvision.datasets import *
from torchvision.transforms import *
import json
from torch.autograd import Variable
import numpy as np
import random

random.seed(1000)


def get_train_loaders(dataset_name, trial_num, config):
    """
    Create training dataloaders.

    dataset_name: name of dataset
    trial_num: trial number dictating known/unknown class split
    config: config file

    returns trainloader, evalloader, testloader, mapping - changes labels from original to known class label
    """
    train_set, val_set, test_set, _ = load_and_transform(
        dataset_name, config, trial_num
    )  # Loads transformed data

    # Open and set the training and validation indices
    with open(f"datasets/{dataset_name}/trainval_idxs.json") as f:
        train_val_idxs = json.load(f)
        train_idxs = train_val_idxs["Train"]
        val_idxs = train_val_idxs["Val"]

    # Open and set the known and unknwon class indices
    with open(f"datasets/{dataset_name}/splits/spl_{trial_num}.json") as f:
        class_splits = json.load(f)
        known_classes = class_splits["Known"]

    # Create train, val and test subsets for the known classes using the indices loaded above.
    train_subset = create_data_subsets(train_set, known_classes, train_idxs)
    val_subset = create_data_subsets(val_set, known_classes, val_idxs)
    known_subset = create_data_subsets(test_set, known_classes)

    # create a mapping from dataset target class number to network known class number
    mapping = create_target_map(known_classes, config["num_classes"])

    batch_size = config["batch_size"]

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["dataloader_workers"],
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        known_subset, batch_size=batch_size, shuffle=True
    )

    return trainloader, valloader, testloader, mapping


def get_eval_loaders(dataset_name, trial_num, config):
    """
    Create evaluation dataloaders.

    dataset_name: name of dataset
    trial_num: trial number dictating known/unknown class split
    config: config file

    returns known_loader, unknown_loader, mapping - changes labels from original to known class label
    """
    if "+" in dataset_name or "All" in dataset_name:
        _, _, test_set, unknown_set = load_and_transform(
            dataset_name, config, trial_num
        )
    else:
        _, _, test_set, _ = load_and_transform(dataset_name, config, trial_num)

    with open("datasets/{}/splits/spl_{}.json".format(dataset_name, trial_num)) as f:
        class_splits = json.load(f)
        known_classes = class_splits["Known"]
        unknown_classes = class_splits["Unknown"]

    known_subset = create_data_subsets(test_set, known_classes)

    if "+" in dataset_name or "All" in dataset_name:
        unknown_subset = create_data_subsets(unknown_set, unknown_classes)
    else:
        unknown_subset = create_data_subsets(test_set, unknown_classes)

    # create a mapping from dataset target class number to network known class number
    mapping = create_target_map(known_classes, config["num_classes"])

    batch_size = config["batch_size"]

    known_loader = torch.utils.data.DataLoader(
        known_subset, batch_size=batch_size, shuffle=False
    )
    unknown_loader = torch.utils.data.DataLoader(
        unknown_subset, batch_size=batch_size, shuffle=False
    )

    return known_loader, unknown_loader, mapping


def get_data_stats(dataset, known_classes):
    """
    Calculates mean and std of data in a dataset.

    dataset: dataset to calculate mean and std of
    known_classes: what classes are known and should be included

    returns means and stds of data, across each colour channel
    """
    try:
        ims = np.asarray(dataset.data)
        try:
            labels = np.asarray(dataset.targets)
        except:
            labels = np.asarray(dataset.labels)

        mask = labels == 1000
        for cl in known_classes:
            mask = mask | (labels == cl)
        known_ims = ims[mask]

        means = []
        stds = []
        if len(np.shape(known_ims)) < 4:
            means += [known_ims.mean() / 255]
            stds += [known_ims.std() / 255]
        else:
            for i in range():
                means += [known_ims[:, :, :, i].mean() / 255]
                stds += [known_ims[:, :, :, i].std() / 255]
    except:
        imloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

        r_data = []
        g_data = []
        b_data = []
        for i, data in enumerate(imloader):
            im, labels = data
            mask = labels == 1000
            for cl in known_classes:
                mask = mask | (labels == cl)
            if torch.sum(mask) == 0:
                continue
            im = im[mask]
            r_data += im[:, 0].detach().tolist()
            g_data += im[:, 1].detach().tolist()
            b_data += im[:, 2].detach().tolist()
        means = [np.mean(r_data), np.mean(g_data), np.mean(b_data)]
        stds = [np.std(r_data), np.std(g_data), np.std(b_data)]
    return means, stds


def create_data_transformations_pipeline(config, trial_num):
    # Parameters for controlling data transformation
    means = config["data_mean"][trial_num]
    stds = config["data_std"][trial_num]
    flip = config["data_transforms"]["flip"]
    rotate = config["data_transforms"]["rotate"]
    scale_min = config["data_transforms"]["scale_min"]

    # Define set of transformations pipeline for train, val and test sets.
    # The train set contains more transformations than val and test sets in order to serve as an augmentation.
    data_transforms = {
        "train": Compose(
            [
                Resize(config["im_size"]),
                RandomResizedCrop(config["im_size"], scale=(scale_min, 1.0)),
                RandomHorizontalFlip(flip),
                RandomRotation(rotate),
                ToTensor(),
                Normalize(means, stds),
            ]
        ),
        "val": Compose([Resize(config["im_size"]), ToTensor(), Normalize(means, stds)]),
        "test": Compose(
            [Resize(config["im_size"]), ToTensor(), Normalize(means, stds)]
        ),
    }
    return data_transforms


def load_and_transform(dataset_name, config, trial_num):
    """
    Load all datasets for training/evaluation.

    dataset_name: name of dataset
    config: config file
    trial_num: trial number dictating known/unknown class split

    returns train_set, val_set, known_set, unknown_set
    """
    # with open("datasets/{}/splits/spl_{}.json".format(dataset_name, trial_num)) as f:
    #     class_splits = json.load(f)
    #     known_classes = class_splits["Known"]

    # Perform data transformation
    data_transforms = create_data_transformations_pipeline(config, trial_num)

    unknown_set = None

    # Perform data transformation for the
    if dataset_name == "MNIST":
        # Load the datasets using torchvision dataset.
        # Set train = True for train and val, and False for test set.
        transformed_train = MNIST(
            "datasets/data", train=True, transform=data_transforms["train"]
        )
        transformed_val = MNIST(
            "datasets/data", train=True, transform=data_transforms["val"]
        )
        transformed_test = MNIST(
            "datasets/data", train=False, transform=data_transforms["test"]
        )
    elif "CIFAR" in dataset_name:
        transformed_train = CIFAR10("datasets/data", transform=data_transforms["train"])
        transformed_val = CIFAR10("datasets/data", transform=data_transforms["val"])
        transformed_test = CIFAR10(
            "datasets/data", train=False, transform=data_transforms["test"]
        )
        if "+" in dataset_name:
            unknown_set = CIFAR100(
                "datasets/data",
                train=False,
                transform=data_transforms["test"],
                download=True,
            )
    elif dataset_name == "SVHN":
        train_set = SVHN(
            "datasets/data", transform=data_transforms["train"], download=True
        )
        val_set = SVHN("datasets/data", transform=data_transforms["val"])
        test_set = SVHN(
            "datasets/data", split="test", transform=data_transforms["test"]
        )
    elif dataset_name == "TinyImageNet":
        train_set = ImageFolder(
            "datasets/data/tiny-imagenet-200/train", transform=data_transforms["train"]
        )
        val_set = ImageFolder(
            "datasets/data/tiny-imagenet-200/train", transform=data_transforms["val"]
        )
        test_set = ImageFolder(
            "datasets/data/tiny-imagenet-200/val", transform=data_transforms["test"]
        )
    else:
        print("Sorry, that dataset has not been implemented.")
        exit()

    return transformed_train, transformed_val, transformed_test, unknown_set


def create_data_subsets(dataset, classes_to_use, idxs_to_use=None):
    """
    Returns dataset subset that satisfies class and idx restraints.
    dataset: torchvision dataset
    classes_to_use: classes that are allowed in the subset (known vs unknown)
    idxs_to_use: image indexes that are allowed in the subset (train vs val, not relevant for test)

    returns torch Subset
    """
    # get class label for dataset images. svhn has different syntax as .labels
    try:
        targets = dataset.targets
    except:
        targets = dataset.labels

    subset_idxs = []
    if idxs_to_use == None:
        for i, lbl in enumerate(targets):
            if lbl in classes_to_use:
                subset_idxs += [i]
    else:
        for class_num in idxs_to_use.keys():
            if int(class_num) in classes_to_use:
                subset_idxs += idxs_to_use[class_num]

    data_subset = torch.utils.data.Subset(dataset, subset_idxs)
    return data_subset


def create_target_map(known_classes, num_classes):
    """
    Creates a mapping from original dataset labels to new 'known class' training label
    known_classes: classes that will be trained with
    num_classes: number of classes the dataset typically has

    returns mapping - a dictionary where mapping[original_class_label] = known_class_label
    """
    mapping = [None for i in range(num_classes)]

    known_classes.sort()
    for i, num in enumerate(known_classes):
        mapping[num] = i

    return mapping
