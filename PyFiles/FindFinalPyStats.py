import pandas as pd
from statistics import median
from sys import argv

absolute_path = argv[1]
path = absolute_path + "/PyResults/"
path_to_save_files = absolute_path + "/PyProcessedResults/"
names = ["acc_py", "scores_py"]
files = [path + name for name in names]
stats = [pd.read_csv(file, header=None)[0].values for file in files]
acc = stats[0]
score = stats[1]
min_acc = min(acc)
median_acc = median(acc)
max_acc = max(acc)
min_score = min(score)
median_score = median(score)
max_score = max(score)
pd.DataFrame([min_acc, median_acc, max_acc]).\
    to_csv(path_to_save_files + "acc_stats", index=False, header=False)
pd.DataFrame([min_score, median_score, max_score]).\
    to_csv(path_to_save_files + "scores_stats", index=False, header=False)