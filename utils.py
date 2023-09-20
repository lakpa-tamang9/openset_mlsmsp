"""
    Helper functions for training and evaluation.

    progress_bar and format_time function was taken from https://github.com/kuangliu/pytorch-cifar which mimics xlua.progress

    Dimity Miller, 2020
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics

try:
    _, term_width = os.popen("stty size", "r").read().split()
    term_width = int(term_width)
except:
    term_width = 84

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def get_standardized_max_logits(logits):
    mean_logits = torch.mean(logits)
    std_dev_logits = torch.std(logits)
    standardized_logits = (logits - mean_logits) / std_dev_logits
    return standardized_logits


def get_max_logits(logits):
    return torch.argmax(logits, dim=1)


def get_softmax_prob(logits):
    loss = F.softmax(logits, dim=0)
    return loss


def get_accuracy(x, gt, use_mls, use_stmls):
    predicted = np.argmax(x, axis=1)
    total = len(gt)
    acc = np.sum(predicted == gt) / total
    return acc


def get_auroc(inData, outData, in_low=True):
    inDataMin = np.min(inData, 1)
    outDataMin = np.min(outData, 1)

    y_hat = np.concatenate((inDataMin, outDataMin))
    y_true = np.concatenate((np.zeros(len(inDataMin)), np.ones(len(outDataMin))))
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_hat, pos_label=in_low)

    return sklearn.metrics.auc(fpr, tpr)


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append("  Step: %s" % format_time(step_time))
    L.append(" | Tot: %s" % format_time(tot_time))
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f
