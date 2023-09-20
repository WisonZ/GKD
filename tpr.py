import os
import sys
import numpy as np
import copy
import time
from prettytable import PrettyTable
import json
# import torch
import matplotlib.pyplot as plt
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap

def find_nearest_fpr(fpr_list):
    id_lst = []
    for FPR in x_labels:
        EPISION = 1e-20
        fpr_nearest = 2000
        fpr_nearest_idx = 0
        for idx, it in enumerate(fpr_list):
            diff_fpr = abs(FPR - float(it + EPISION))
            if diff_fpr < fpr_nearest:
                fpr_nearest = diff_fpr
                fpr_nearest_idx = idx
        id_lst.append(fpr_nearest_idx)
            #
        #

    return fpr_nearest_idx, fpr_list[fpr_nearest_idx],id_lst

def load_json_result_file(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)
# print(torch.__version__)
x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
tpr_fpr_row = []
tpr_fpr_row.append("meg")
result_json_file = '/data1/megaface/devkit/experiments/cmc_facescrub_megaface_r100ii_1000000_1.json'
result_dict = load_json_result_file(result_json_file)
res_rank1 = result_dict["cmc"][1][0]
idx, fpr,id_lst = find_nearest_fpr(result_dict["roc"][0])
colours = sample_colours_from_colourmap(1, 'Set2')
fig = plt.figure()
plt.plot(result_dict["roc"][0],
             result_dict["roc"][1],
             color=colours[0],
             lw=1,
             label=('auc'
                   ))
plt.xlim([10 ** -6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels)
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on meg')
plt.legend(loc="lower right")
fig.savefig(os.path.join('./meg.pdf'))
exit()
res_1vs1 = result_dict["roc"][1][idx]
_results = {}
_results["rank1"] = round(res_rank1 * 100, 4)
_results["1vs1_fpr"] = format(fpr, ".1e")
_results["1vs1"] = round(res_1vs1 * 100, 4)
print(_results["rank1"])
print(_results["1vs1_fpr"] )
print(_results["1vs1"] )
for id in id_lst:
    print(round(result_dict["roc"][1][id] * 100, 4))
    tpr_fpr_row.append('%.2f' % (round(result_dict["roc"][1][id] * 100, 4)))
tpr_fpr_table.add_row(tpr_fpr_row)
print(tpr_fpr_table)

