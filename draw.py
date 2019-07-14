import pandas as pd
import numpy as np
import constants as c
from sklearn import metrics
import matplotlib.pyplot as plt

def evaluate_metrics(y_true, y_pres):
    plt.figure()
    for y_pre in y_pres:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pre, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr)
    
    # plt.plot(np.arange(1, 0, -0.01), np.arange(0, 1, 0.01))
    plt.legend(['seresnet','seresnet-nodrop'])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title(f'ROC curve, AUC score={auc}')
    plt.show()

    threshold_index = np.argmin(abs(1-tpr - fpr))
    threshold = thresholds[threshold_index]
    eer = ((1-tpr)[threshold_index]+fpr[threshold_index])/2
    print(eer)
    auc_score = metrics.roc_auc_score(y_true, y_pre, average='macro')

    y_pro = [1 if x > threshold else 0 for x in y_pre]
    acc = metrics.accuracy_score(y_true, y_pro)
    prauc = metrics.average_precision_score(y_true, y_pro, average='macro')
    return y_pro, eer, prauc, acc, auc_score


def speaker_verification(distances, ismember_true):
    for distance in distances:
        score_index = distances.argmax(axis=0)
        distance_max = distances.max(axis=0)
        distance_max = (distance_max + 1) / 2
        y_pro, eer, prauc, acc, auc_score = evaluate_metrics(
            ismember_true, distance_max)

    print(f'eer={eer}\t prauc={prauc} \t acc={acc}\t auc_score={auc_score}\t')
    return y_pro


if __name__ == "__main__":
    distance1 = np.load('./npys/perfect.npy')
    distance2 = np.load('./npys/perfectnew.npy')
    distance = (distance1,distance2)
    df = pd.read_csv(c.ANNONATION_FILE)
    ismember_true = list(map(int,df["Ismember"]))
    ismember_pre = speaker_verification(distance, ismember_true)