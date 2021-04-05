import numpy as np
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

with open("config.yml") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


def get_best_threshold(y_true, pred_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, pred_probs)
    fscore = (2 * precision * recall) / (precision + recall)

    # Find the optimal threshold
    index = np.argmax(fscore)
    thresholdOpt = round(thresholds[index], ndigits=4)
    fscoreOpt = round(fscore[index], ndigits=4)
    recallOpt = round(recall[index], ndigits=4)
    precisionOpt = round(precision[index], ndigits=4)
    print("Best Threshold: {} with F-Score: {}".format(thresholdOpt, fscoreOpt))
    print("Recall: {}, Precision: {}".format(recallOpt, precisionOpt))

    return thresholdOpt


def plot_pr_curve(y_true, pred_probs):
    pr, rec, thr = precision_recall_curve(y_true, pred_probs)
    auc = roc_auc_score(y_true, pred_probs)

    plt.figure(figsize=(7, 7))
    plt.plot(rec, pr)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"AUC: {auc}")

    plt.show()

    return rec, pr


def get_report(y_true, pred_probs):
    threshold = get_best_threshold(y_true, pred_probs)

    class_predictions = [int(prob > threshold) for prob in pred_probs]
    print(
        "\n(With best threshold) Accuracy: %.2f, F1: %.2f"
        % (
            accuracy_score(y_true, class_predictions),
            f1_score(y_true, class_predictions),
        )
    )

    return plot_pr_curve(y_true, pred_probs)
