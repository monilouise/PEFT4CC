from river import metrics

def calculate_rolling_roc_auc(pred_prob, true_label):
    metric = metrics.RollingROCAUC()
    for yt, yp in zip(true_label, pred_prob):
        if len(yp.shape) == 1:
            yp = yp[0]
        metric.update(yt, yp)
    return metric