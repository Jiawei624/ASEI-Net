"""Range-based detection metrics and volume-under-surface calculations."""
from sklearn import metrics
import numpy as np
import math


class metricor:
    def __init__(
        self, a=1, probability=True, bias="flat",
    ):
        self.a = a
        self.probability = probability
        self.bias = bias

    def detect_model(
        self, model, label, contamination=0.1, window=100, is_A=False, is_threshold=True
    ):
        if is_threshold:
            score = self.scale_threshold(
                model.decision_scores_, model._mu, model._sigma
            )
        else:
            score = self.scale_contamination(
                model.decision_scores_, contamination=contamination
            )
        if is_A is False:
            scoreX = np.zeros(len(score) + window)
            scoreX[
                math.ceil(window / 2) : len(score) + window - math.floor(window / 2)
            ] = score
        else:
            scoreX = score

        self.score_ = scoreX
        L = self.metric(label, scoreX)
        return L

    def labels_conv(self, preds):
        """return indices of predicted anomaly
        """

        index = np.where(preds >= 0.5)
        return index[0]

    def labels_conv_binary(self, preds):
        """return predicted label
        """
        p = np.zeros(len(preds))
        index = np.where(preds >= 0.5)
        p[index[0]] = 1
        return p

    def w(self, AnomalyRange, p):
        MyValue = 0
        MaxValue = 0
        start = AnomalyRange[0]
        AnomalyLength = AnomalyRange[1] - AnomalyRange[0] + 1
        for i in range(start, start + AnomalyLength):
            bi = self.b(i, AnomalyLength)
            MaxValue += bi
            if i in p:
                MyValue += bi
        return MyValue / MaxValue

    def Cardinality_factor(self, Anomolyrange, Prange):
        score = 0
        start = Anomolyrange[0]
        end = Anomolyrange[1]
        for i in Prange:
            if i[0] >= start and i[0] <= end:
                score += 1
            elif start >= i[0] and start <= i[1]:
                score += 1
            elif end >= i[0] and end <= i[1]:
                score += 1
            elif start >= i[0] and end <= i[1]:
                score += 1
        if score == 0:
            return 0
        else:
            return 1 / score

    def b(self, i, length):
        bias = self.bias
        if bias == "flat":
            return 1
        elif bias == "front-end bias":
            return length - i + 1
        elif bias == "back-end bias":
            return i
        else:
            if i <= length / 2:
                return i
            else:
                return length - i + 1

    def scale_threshold(self, score, score_mu, score_sigma):
        return (score >= (score_mu + 3 * score_sigma)).astype(int)

    def metric_new(self, label, score, plot_ROC=False, alpha=0.2):
        """input:
               Real labels and anomaly score in prediction
            
           output:
               AUC, 
               Precision, 
               Recall, 
               F-score, 
               Range-precision, 
               Range-recall, 
               Range-Fscore, 
               Precison@k, 
             
            k is chosen to be # of outliers in real labels
        """
        if np.sum(label) == 0:
            print(
                "All labels are 0. Label must have groud truth value for calculating AUC score."
            )
            return None

        if np.isnan(score).any() or score is None:
            print("Score must not be none.")
            return None

        # area under curve
        both_classes = len(np.unique(label)) == 2
        auc = metrics.roc_auc_score(label, score) if both_classes else float("nan")
        # plor ROC curve
        if plot_ROC:
            if both_classes:
                fpr, tpr, thresholds = metrics.roc_curve(label, score)
            else:
                fpr = tpr = np.array([np.nan])
            # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
            # display.plot()

        # precision, recall, F
        preds = score
        # preds = score > (np.mean(score) + 3 * np.std(score))
        Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
            label, preds, labels=[1], zero_division=0
        )
        precision = Precision[0]
        recall = Recall[0]
        f = F[0]

        # range anomaly
        Rrecall, ExistenceReward, OverlapReward = self.range_recall_new(
            label, preds, alpha
        )
        Rprecision = self.range_recall_new(preds, label, 0)[0]

        if Rprecision + Rrecall == 0:
            Rf = 0
        else:
            Rf = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)

        # top-k
        k = int(np.sum(label))
        threshold = np.percentile(score, 100 * (1 - k / len(label)))

        # precision_at_k = metrics.top_k_accuracy_score(label, score, k)
        p_at_k = np.where(preds > threshold)[0]
        TP_at_k = sum(label[p_at_k])
        precision_at_k = TP_at_k / k

        L = [
            auc,
            precision,
            recall,
            f,
            Rrecall,
            ExistenceReward,
            OverlapReward,
            Rprecision,
            Rf,
            precision_at_k,
        ]
        if plot_ROC:
            return L, fpr, tpr
        return L

    def metric_PR(self, label, score):
        precision, recall, thresholds = metrics.precision_recall_curve(label, score)
        # plt.figure()
        # disp = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
        # disp.plot()
        AP = metrics.average_precision_score(label, score)
        return precision, recall, AP

    def range_recall_new(self, labels, preds, alpha):

        p = np.where(preds == 1)[0]  # positions of predicted label==1
        range_pred = self.range_convers_new(preds)
        range_label = self.range_convers_new(labels)

        Nr = len(range_label)  # total # of real anomaly segments

        if not Nr:
            return 0, 0, 0
        if self.bias == "flat":
            bounds = np.asarray(range_label, dtype=np.int64)
            hits = np.searchsorted(p, bounds[:, 1], side="right") - np.searchsorted(p, bounds[:, 0])
            existence = np.count_nonzero(hits)
            overlap = 0.
            if range_pred:
                predicted = np.asarray(range_pred, dtype=np.int64)
                cardinality = (np.searchsorted(predicted[:, 0], bounds[:, 1], side="right")
                               - np.searchsorted(predicted[:, 1], bounds[:, 0]))
                overlap = np.sum(np.divide(hits / (bounds[:, 1] - bounds[:, 0] + 1), cardinality,
                                           out=np.zeros(Nr, dtype=float), where=cardinality > 0))
            return (alpha * existence + (1 - alpha) * overlap) / Nr, existence / Nr, overlap / Nr

        ExistenceReward = self.existence_reward(range_label, p)

        OverlapReward = 0
        for i in range_label:
            OverlapReward += self.w(i, p) * self.Cardinality_factor(i, range_pred)

        score = alpha * ExistenceReward + (1 - alpha) * OverlapReward
        if Nr != 0:
            return score / Nr, ExistenceReward / Nr, OverlapReward / Nr
        else:
            return 0, 0, 0

    def range_convers_new(self, label):
        """
        input: arrays of binary values 
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        """
        edges = np.diff(np.r_[0, (np.asarray(label) != 0).astype(np.int8), 0])
        return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))

    def existence_reward(self, labels, preds):
        """
        labels: list of ordered pair 
        preds predicted data
        """

        score = 0
        for i in labels:
            if np.sum(np.multiply(preds <= i[1], preds >= i[0])) > 0:
                score += 1
        return score

    def num_nonzero_segments(self, x):
        count = 0
        if x[0] > 0:
            count += 1
        for i in range(1, len(x)):
            if x[i] > 0 and x[i - 1] == 0:
                count += 1
        return count

    def extend_postive_range(self, x, window=5):
        # Preserve TAB commit 5a7e6f8, including its inclusive-end extension.
        # Changing e to e+1 changes the external metric protocol.
        label = x.copy().astype(float)
        L = self.range_convers_new(label)  # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            x1 = np.arange(e, min(e + window // 2, length))
            label[x1] += np.sqrt(1 - (x1 - e) / (window))

            x2 = np.arange(max(s - window // 2, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / (window))

        label = np.minimum(np.ones(length), label)
        return label

    def extend_postive_range_individual(self, x, percentage=0.2):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)  # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            l0 = int((e - s + 1) * percentage)

            x1 = np.arange(e, min(e + l0, length))
            label[x1] += np.sqrt(1 - (x1 - e) / (2 * l0))

            x2 = np.arange(max(s - l0, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / (2 * l0))

        label = np.minimum(np.ones(length), label)
        return label

    def TPR_FPR_RangeAUC(self, labels, pred, P, L):
        product = labels * pred 

        TP = np.sum(product)

        # recall = min(TP/P,1)
        P_new = (P + np.sum(labels)) / 2  # so TPR is neither large nor small
        # P_new = np.sum(labels)
        recall = min(TP / P_new, 1)
        # recall = TP/np.sum(labels)
        # print('recall '+str(recall))

        existence = 0
        for seg in L:
            if np.sum(product[seg[0] : (seg[1] + 1)]) > 0:
                existence += 1

        existence_ratio = existence / len(L)
        # print(existence_ratio)

        # TPR_RangeAUC = np.sqrt(recall*existence_ratio)
        # print(existence_ratio)
        TPR_RangeAUC = recall * existence_ratio

        FP = np.sum(pred) - TP
        # TN = np.sum((1-pred) * (1-labels))

        # FPR_RangeAUC = FP/(FP+TN)
        N_new = len(labels) - P_new
        FPR_RangeAUC = FP / N_new

        Precision_RangeAUC = TP / np.sum(pred)

        return TPR_RangeAUC, FPR_RangeAUC, Precision_RangeAUC

    def RangeAUC(
        self, labels, score, window=0, percentage=0, plot_ROC=False, AUC_type="window"
    ):
        ranked = self._ranked_thresholds(score)
        P = np.sum(labels)
        if AUC_type == "window":
            labels = self.extend_postive_range(labels, window=window)
        else:
            labels = self.extend_postive_range_individual(labels, percentage=percentage)

        AUC_range, AP_range, fpr, tpr, prec = self._ranked_range_curve(labels, P, ranked)
        if plot_ROC:
            return AUC_range, AP_range, fpr, tpr, prec
        return AUC_range

    @staticmethod
    def _ranked_thresholds(score):
        """The same 250 thresholds and >= tie rule as TAB, sorted only once."""
        score = np.asarray(score)
        order = np.argsort(-score, kind="stable")
        descending = score[order]
        thresholds = descending[np.linspace(0, len(score) - 1, 250).astype(int)]
        counts = np.searchsorted(-descending, -thresholds, side="right")
        ranks = np.empty(len(score) + 1, dtype=np.int64)
        ranks[order] = np.arange(len(score))
        ranks[-1] = len(score)  # sentinel for a segment ending at the last point
        return order, counts, ranks

    def _ranked_range_curve(self, labels, positives, ranked):
        """Prefix sums replace 250 complete scans without changing the curve."""
        order, counts, ranks = ranked
        labels = np.asarray(labels, dtype=np.float64)
        tp = np.cumsum(labels[order], dtype=np.float64)[counts - 1]
        segments = np.asarray(self.range_convers_new(labels), dtype=np.int64)
        if not len(segments):
            raise ValueError("Range AUC requires an anomaly segment.")
        edges = np.column_stack((segments[:, 0], segments[:, 1] + 1)).ravel()
        first_hits = np.sort(np.minimum.reduceat(ranks, edges)[::2])
        existence = np.searchsorted(first_hits, counts, side="left") / len(segments)
        p_new = (positives + labels.sum()) / 2
        tpr = np.r_[0., np.minimum(tp / p_new, 1.) * existence, 1.]
        fpr = np.r_[0., (counts - tp) / (len(labels) - p_new), 1.]
        precision = np.r_[1., tp / counts]
        auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
        ap = np.sum(np.diff(tpr[:-1]) * (precision[1:] + precision[:-1]) / 2)
        return auc, ap, fpr, tpr, precision

    # TPR_FPR_window
    def RangeAUC_volume(self, labels_original, score, windowSize, workers=1):
        from ..parallel import ordered_prefetch
        ranked = self._ranked_thresholds(score)

        tpr_3d = []
        fpr_3d = []
        prec_3d = []

        auc_3d = []
        ap_3d = []

        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)

        def calculate(window):
            labels = self.extend_postive_range(labels_original, window)
            return self._ranked_range_curve(labels, P, ranked)
        # Ordered collection retains the original integration and sum order.
        with ordered_prefetch(calculate, window_3d, workers=workers if workers > 1 else 0,
                              depth=max(1, 2 * workers)) as pending:
            for AUC_range, AP_range, fpr, tpr, prec in pending:
                tpr_3d.append(tpr)
                fpr_3d.append(fpr)
                prec_3d.append(prec)
                auc_3d.append(AUC_range)
                ap_3d.append(AP_range)

        return (
            tpr_3d,
            fpr_3d,
            prec_3d,
            window_3d,
            sum(auc_3d) / len(window_3d),
            sum(ap_3d) / len(window_3d),
        )


def generate_curve(label, score, slidingWindow, workers=1):
    (
        tpr_3d,
        fpr_3d,
        prec_3d,
        window_3d,
        avg_auc_3d,
        avg_ap_3d,
    ) = metricor().RangeAUC_volume(
        labels_original=label, score=score, windowSize=1 * slidingWindow, workers=workers
    )

    X = np.array(tpr_3d).reshape(1, -1).ravel()
    X_ap = np.array(tpr_3d)[:, :-1].reshape(1, -1).ravel()
    Y = np.array(fpr_3d).reshape(1, -1).ravel()
    W = np.array(prec_3d).reshape(1, -1).ravel()
    Z = np.repeat(window_3d, len(tpr_3d[0]))
    Z_ap = np.repeat(window_3d, len(tpr_3d[0]) - 1)

    return Y, Z, X, X_ap, W, Z_ap, avg_auc_3d, avg_ap_3d
