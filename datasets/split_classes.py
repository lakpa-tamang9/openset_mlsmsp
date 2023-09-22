import json
import random
import os


# MNSIT : 10 classes
# Tinyimagenet : 200 classes
def split_classes(num_knowns, total_class_num):
    class_idxs = [class_id for class_id in range(total_class_num)]
    random.shuffle(class_idxs)
    known_classes = class_idxs[:num_knowns]
    unknown_classes = class_idxs[num_knowns:]
    return {"Known": known_classes, "Unknown": unknown_classes}


def save_splits(dataset_name, num_knowns, total_class_num):
    CWD = os.getcwd()
    split_path = f"{CWD}/datasets/{dataset_name}/splits"
    if not os.path.exists(split_path):
        os.makedirs(split_path)
    for i in range(5):
        dump_data = split_classes(num_knowns, total_class_num)
        with open(f"{split_path}/spl_{i}.json", "w") as f:
            json.dump(dump_data, f)


if __name__ == "__main__":
    save_splits("TinyImageNet", num_knowns=20, total_class_num=200)
    print("Class split complete and saved to respective directory!")
