from sys import argv
import pandas as pd
from PyFiles import Tools

absolute_path = argv[1]
path = absolute_path + "/RResults/"
names = ["pred_chi", "pred_w"]
saved_file_names = ["chi", "w"]
files = [path + name for name in names]
yt = pd.read_csv(path + "yt", header=None)[0].values
for i, name in enumerate(files):
    predicted = pd.read_csv(files[i], header=None)[0].values
    accuracy = Tools.accuracy(yt, predicted)
    score = Tools.MMC(yt, predicted)
    pd.DataFrame([accuracy]). \
        to_csv(path + "acc_" + saved_file_names[i], index=False, header=False, mode="a")
    pd.DataFrame([score]). \
        to_csv(path + "scores_" + saved_file_names[i], index=False, header=False, mode="a")