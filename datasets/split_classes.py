import json
import random
import os


# MNSIT : 10 classes
def split_classes(num_knowns):
    class_idxs = [class_id for class_id in range(10)]
    random.shuffle(class_idxs)
    known_classes = class_idxs[:num_knowns]
    unknown_classes = class_idxs[num_knowns:]
    return {"Known": known_classes, "Unknown": unknown_classes}


def save_splits(dataset_name, num_knowns):
    CWD = os.getcwd()
    split_path = f"{CWD}/datasets/{dataset_name}/splits"
    if not os.path.exists(split_path):
        os.makedirs(split_path)
    for i in range(5):
        dump_data = split_classes(num_knowns)
        with open(f"{split_path}/spl_{i}.json", "w") as f:
            json.dump(dump_data, f)


# if __name__ == "__main__":
#     save_splits("MNIST", num_knowns=4)
#     print("Class split complete and saved to respective directory!")
