from sys import argv
import pandas as pd
from statistics import median

absolute_path = argv[1]
path = absolute_path + "/RResults/"
path_to_save_files = absolute_path + "/RProcessedResults/"
acc_file_names = [path + file for file in ["acc_chi", "acc_w"]]
score_file_names = [path + file for file in ["scores_chi", "scores_w"]]
acc_file_names_to_save = [path_to_save_files + file for file in ["acc_chi_stats", "acc_w_stats"]]
score_file_names_to_save = [path_to_save_files + file for file in ["scores_chi_stats", "scores_w_stats"]]
for acc_prev, scores_prev, acc_next, scores_next in zip(acc_file_names, score_file_names, acc_file_names_to_save, score_file_names_to_save):
    acc = pd.read_csv(acc_prev, header=None)[0].values
    score = pd.read_csv(scores_prev, header=None)[0].values
    min_acc = min(acc)
    median_acc = median(acc)
    max_acc = max(acc)
    min_score = min(score)
    median_score = median(score)
    max_score = max(score)
    pd.DataFrame([min_acc, median_acc, max_acc]).\
        to_csv(acc_next, index=False, header=False)
    pd.DataFrame([min_score, median_score, max_score]). \
        to_csv(scores_next, index=False, header=False)