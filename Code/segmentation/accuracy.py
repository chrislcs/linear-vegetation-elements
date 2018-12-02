# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
from shapely.ops import cascaded_union


def matthews_correlation(tp, tn, fp, fn):
    """
    """
    mcc = ((tp*tn)-(fp*fn))/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return mcc


def overall_accuracy(tp, tn, fp, fn):
    """
    """
    acc = (tp+tn)/(tp+tn+fp+fn)
    return acc


def recall(tp, fp):
    # user's accuracy
    return tp / (tp + fp)


def precision(tp, fn):
    # producer's accuracy
    return tp / (tp + fn)


def f1score(rec, prec):
    f1 = 2 / ((1/rec) + (1/prec))
    return f1


def spatial_assessment(manual_shapes_linear,
                       manual_shapes_nonlinear,
                       automated_shapes):
    """
    """
    linear = cascaded_union(manual_shapes_linear)
    if not linear.is_valid:
        linear = linear.buffer(0)
    nonlinear = cascaded_union(manual_shapes_nonlinear)
    if not nonlinear.is_valid:
        nonlinear = nonlinear.buffer(0)
    automated = cascaded_union(automated_shapes)
    if not automated.is_valid:
        automated = automated.buffer(0)

    true_positive = linear.intersection(automated)
    false_positive = automated.difference(linear)
    true_negative = nonlinear.difference(automated)
    false_negative = linear.difference(automated)

    return true_positive, false_positive, true_negative, false_negative


def numerical_assessment(true_positive, false_positive,
                         true_negative, false_negative):
    """
    """

    tp = true_positive.area
    tn = true_negative.area
    fp = false_positive.area
    fn = false_negative.area

    mcc = matthews_correlation(tp, tn, fp, fn)
    acc = overall_accuracy(tp, tn, fp, fn)
    rec = recall(tp, fp)
    prec = precision(tp, fn)
    f1 = f1score(rec, prec)

    return {'true_positive': tp,
            'false_positive': fp,
            'true_negative': tn,
            'false_negative': fn,
            'acc': acc,
            'mcc': mcc,
            'rec': rec,
            'prec': prec,
            'f1': f1}
