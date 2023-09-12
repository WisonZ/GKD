#!/usr/bin/env python
# coding: utf-8

"""
`find_score` and `compute_roc` is copied from ws
"""

import numpy as np
import os.path as osp
import os
from math import sqrt
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Iterable
import pdb

def find_score(far, vr, target=1e-4):
    # far is an ordered array, find the index of far whose element is closest to target, and return vr[index]
    # assert isinstance(far, list)
    l = 0
    u = far.size - 1
    while u - l > 1:
        mid = round((l + u) / 2)
        # print far[mid]
        if far[mid] == target:
            return vr[mid]
        elif far[mid] < target:
            u = mid
        else:
            l = mid
    # Actually, either array[l] or both[u] is not equal to target, so I do interpolation here.
    # print (vr[l] + vr[u]) / 2.0
    if far[l] / target >= 8:  # cannot find points that's close enough to target.
        return 0.0
    return (vr[l] + vr[u]) / 2.0, (l+u)/2


def compute_roc(score, label, num_thresholds=1000):
    pos_dist = score[label == 1]
    neg_dist = score[label == 0]
    num_pos_samples = pos_dist.size
    num_neg_samples = neg_dist.size
    data_max = np.max(score)
    data_min = np.min(score)
    #pdb.set_trace()
    unit = (data_max - data_min) * 1.0 / num_thresholds
    threshold = data_min + (data_max - data_min) * np.array(range(1, num_thresholds + 1)) / num_thresholds
    new_interval = threshold - unit / 2.0 + 2e-6
    new_interval = np.append(new_interval, np.array(new_interval[-1] + unit))
    P = np.triu(np.ones(num_thresholds))

    pos_hist, dummy = np.histogram(pos_dist, new_interval)
    neg_hist, dummy2 = np.histogram(neg_dist, new_interval)
    pos_mat = pos_hist[:, np.newaxis]
    neg_mat = neg_hist[:, np.newaxis]

    assert pos_hist.size == neg_hist.size == num_thresholds
    far = np.dot(P, neg_mat) / num_neg_samples
    far = np.squeeze(far)
    vr = np.dot(P, pos_mat) / num_pos_samples
    vr = np.squeeze(vr)
    return far, vr


def draw_hist(y, colors=None, title='', wfp=None):
    plt.close()
    x = np.arange(len(y))
    if colors is None:
        plt.bar(x, y)
    else:
        plt.bar(x, y, color=colors)

    plt.xlabel('#Rank')
    plt.ylabel('Score')

    xticks = ['GT'] + range(1, 1 + len(y))
    plt.xticks(x, xticks)
    plt.title(title)

    if wfp is None:
        plt.show()
    else:
        plt.savefig(wfp, dpi=100, bbox_inches='tight', pad_inches=0.1)


def draw_hist(score, title='hist', wfp=None, plt_flg=False):
    """Draw hist demo for reference"""
    bins = np.linspace(-1, 1, 401)
    plt.close()
    plt.style.use('bmh')

    plt.hist(score, bins=bins, normed=1)

    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.grid(True)

    if wfp:
        plt.savefig(wfp, dpi=450)
        print('Wrote to {}'.format(wfp))
    if plt_flg:
        plt.show()


def draw_score_dist(score, label, title='score hist', wfp=None, plt_flg=False, draw_pos_flg=True, draw_neg_flg=True):
    """Draw positive score and negative score distribution"""
    bins = np.linspace(np.min(score), np.max(score), 401)
    plt.close()
    plt.style.use('bmh')

    assert (draw_neg_flg is True or draw_pos_flg is True)
    if draw_neg_flg:
        plt.hist(score[label == 0], bins=bins, normed=1, alpha=.7)
    if draw_pos_flg:
        plt.hist(score[label == 1], bins=bins, normed=1, alpha=.7)

    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.grid(True)

    if wfp:
        plt.savefig(wfp, dpi=450)
        print('Wrote to {}'.format(wfp))
    if plt_flg:
        plt.show()




def draw_dist(score, title='score hist', wfp=None, plt_flg=False, min=None, max=None):
    """Draw positive score and negative score distribution"""
    if min is not None:
        mins = min(score)
    if max is not None:
        maxs = max(score)

    bins = np.linspace(mins, maxs, 401)
    plt.close()
    plt.style.use('bmh')

    plt.hist(score, bins=bins, normed=1, alpha=.7)

    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.grid(True)

    if wfp:
        plt.savefig(wfp, dpi=450)
        print('Wrote to {}'.format(wfp))
    if plt_flg:
        plt.show()


def draw_score_dists(scores, label, title='score hist', wfp=None, plt_flg=False, draw_pos_flg=True, draw_neg_flg=True):
    """Draw positive score and negative score distribution: multiple score version"""
    bins = np.linspace(-1, 1, 401)
    plt.close()
    plt.style.use('bmh')

    assert (draw_neg_flg is True or draw_pos_flg is True)
    colors = ['g', 'r']
    if not isinstance(scores, Iterable):
        scores = [scores]
    for i in range(len(scores)):
        if draw_neg_flg:
            plt.hist(scores[i][label == 0], bins=bins, normed=1, alpha=.65, color=colors[i])
        if draw_pos_flg:
            plt.hist(scores[i][label == 1], bins=bins, normed=1, alpha=.65, color=colors[i])

    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.grid(True)

    if wfp:
        plt.savefig(wfp, dpi=450)
        print('Wrote to {}'.format(wfp))
    if plt_flg:
        plt.show()

def save_feat(feat, filename):
    with open(filename, 'w') as outfile:
        assert feat.size > 0
        for i in range(feat.size):
            outfile.write(str(feat[i]) + '\n')


def load_feat(filename):
    # feat = sio.loadmat(osp.join(feat_dir, k + '.mat')).get('feature').ravel()
    fid = open(filename, 'r')
    feat = np.array(fid.readlines(), dtype=float)
    fid.close()
    # return feat / np.sqrt(feat.dot(feat))
    return feat

def main():
    pass


if __name__ == '__main__':
    main()
