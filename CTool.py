import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

def rm_n_mkdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

class CToolMetrics:
    def CaculateFpr(self, y_true, y_pred):
        fp = 0
        sum = 0
        for i in range(len(y_true)):
            if y_true[i] == 0:
                sum += 1
                if y_pred[i] == 1:
                    fp += 1
        if sum == 0:
            return "NaN"
        return fp/sum

    def CaculateNpv(self, y_true, y_pred):
        tn = 0
        sum = 0
        for i in range(len(y_pred)):
            if y_pred[i] == 0:
                sum += 1
                if y_true[i] == 0:
                    tn += 1
        if sum == 0:
            return "NaN"
        return tn/sum

    def CaculateTnr(self, y_true, y_pred):
        fp = 0
        sum = 0
        for i in range(len(y_true)):
            if y_true[i] == 0:
                sum += 1
                if y_pred[i] == 1:
                    fp += 1
        if sum == 0:
            return "NaN"
        return 1 - fp/sum


def DrawValAcc(hist, num_epochs, save_path):
    plt.figure()
    ohist = [h.cpu().numpy() for h in hist]
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, num_epochs + 1), ohist, label="Pretrained")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(save_path, 'val_acc.png'), bbox_inches='tight')

def DrawTrainValLoss(loss, num_epochs, save_path):
    plt.figure()
    plt.title("Train Loss and Val Loss")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1, num_epochs + 1), loss['train'], 'b', label="Train Loss")
    plt.plot(range(1, num_epochs + 1), loss['val'], 'g', label="Val Loss")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(save_path, 'loss.png'), bbox_inches='tight')

def DrawROC(save_path, y_test, y_pre_pro):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pre_pro)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    # 基于TPR-FPR计算计算约登指数，最大约登指数对应的分类阈值即最佳分类阈值
    maxindex = (true_positive_rate - false_positive_rate).tolist().index(max(true_positive_rate - false_positive_rate))
    print("最佳分类阈值", thresholds[maxindex])

    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.scatter(false_positive_rate[maxindex], true_positive_rate[maxindex], c="red", s=30)
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    # plt.show()
    plt.savefig(os.path.join(save_path, 'roc.png'), bbox_inches='tight')