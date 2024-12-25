import CTool
from CTool import CToolMetrics
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

def CFunctionCaculateResult(save_path, y_test, y_pred, y_pre_pro):
    # 绘制ROC曲线
    CTool.DrawROC(save_path, y_test, y_pre_pro)
    # 计算各项指标
    metrics = CToolMetrics()
    result = {}
    result['auc'] = roc_auc_score(y_test, y_pre_pro)
    result['acc'] = accuracy_score(y_test, y_pred)
    result['tpr'] = recall_score(y_test, y_pred)  # 也是sensity
    result['fpr'] = metrics.CaculateFpr(y_test, y_pred)
    result['ppv'] = precision_score(y_test, y_pred)
    result['npv'] = metrics.CaculateNpv(y_test, y_pred)

    return result