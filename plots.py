import matplotlib.pyplot as plt
import json
import numpy as np

with open("./outputs/eval_dict_mls.json") as f1, open(
    "./outputs/eval_dict_msp.json"
) as f2, open("./outputs/eval_dict_stmls.json") as f3:
    mls = json.load(f1)["CIFAR10"]
    msp = json.load(f2)["CIFAR10"]
    stmls = json.load(f3)["CIFAR10"]

acc_mls, auc_mls = [elem[0] for elem in mls], [elem[1] for elem in mls]
acc_msp, auc_msp = [elem[0] for elem in msp], [elem[1] for elem in msp]
acc_stmls, auc_stmls = [elem[0] for elem in stmls], [elem[1] for elem in stmls]

# Plotting auc

categories = [i for i in range(6)]
subcategories = ["MSP", "MLS", "STMLS"]

values = np.transpose(np.array([auc_msp, auc_mls, auc_stmls]))
avg_values = np.mean(values, axis=0).reshape(1, 3)

new_values = np.vstack((values, avg_values))
# Create grouped bar graph
num_categories = len(categories)
num_subcategories = len(subcategories)
bar_width = 0.2  # Width of each bar
index = np.arange(num_categories)  # X location for groups

# Plotting
fig, ax = plt.subplots()

for i in range(num_subcategories):
    ax.bar(
        index + i * bar_width, new_values[:, i], bar_width, label=f"{subcategories[i]}"
    )

ticklabels = categories[:-1]
ticklabels.append("Average")
# Add labels and title
ax.set_xlabel("Trials")
ax.set_ylabel("Area Under the ROC")
ax.set_title("AUC scores for CIFAR10")
ax.set_xticks(index + bar_width)
ax.set_xticklabels(ticklabels)
ax.legend(loc="lower left", fontsize="small")

# Show the plot
plt.savefig("./outputs/bargraph.png")
