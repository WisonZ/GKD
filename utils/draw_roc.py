#!/usr/bin/env python3
# coding: utf-8


from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def _roc_curve(far, vr, plt_flg=True, wfp=None, name='', i=None):
    # plt.style.use('bmh')

    # set font
    # plt.rcParams.update({'font.size': 12})
    plt.rc('font', size=12)
    plt.rc('axes', titlesize=12)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)
    plt.rc('figure', titlesize=12)

    vr = 100 * vr
    plt.semilogx(far, vr, label=name, color='b', linewidth=1.5, alpha=0.9)
    plt.xlim(1e-2, 1e0)
    plt.ylim(0, 100)

    plt.gca().yaxis.grid(True, linestyle='dotted')
    plt.gca().xaxis.grid(True, linestyle='dotted')
    # plt.grid(True)

    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('Truth Positive Rate (TPR, %)')

    # plt.title('#case {}: ROC Curve'.format(i))
    plt.legend()
    plt.savefig('roc.jpg')
    if wfp:
        plt.savefig(wfp, dpi=600)
    if plt_flg:
        plt.show()

    plt.close()


def calc_far_vr(score, label, num_thresholds=2000, return_interval_flg=False, reversed_flg=False):
    """The fast version of calculate roc
    :param score: \in [-1, 1]
    :param label: np.bool type
    :param num_thresholds: the larger, the more accurate
    :return: FAR and VR vector
    1e-7, 1e-8 is not very stable when the sample is small
    """
    assert score.size == label.size
    score_pos = score[label == 0]  # equal to label == True
    score_neg = score[label == 1]

    score_max, score_min = np.max(score), np.min(score)
    unit = (score_max - score_min) / num_thresholds
    interval = score_min - unit / 2. + unit * np.arange(1, num_thresholds + 2)
    U = np.triu(np.ones(num_thresholds))

    score_pos_hist, _ = np.histogram(score_pos, interval)
    score_neg_hist, _ = np.histogram(score_neg, interval)

    vr = U.dot(score_pos_hist).ravel() / score_pos.size
    far = U.dot(score_neg_hist).ravel() / score_neg.size

    if reversed_flg:
        far = far[::-1]
        vr = vr[::-1]
    if not return_interval_flg:
        return far, vr
    return far, vr, interval


def main():
    score = None
    label = None

    far, vr = calc_far_vr(score, label)
    _roc_curve(far, vr, name='brain', wfp='roc.jpg')


if __name__ == '__main__':
    main()
