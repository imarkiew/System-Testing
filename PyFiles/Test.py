from PyFiles import Tools
import pandas as pd
from sys import argv

absolute_path = argv[1]
divided_sets_path = absolute_path + "/DividedSets/"
Xx = pd.read_csv(divided_sets_path + "Xx", header=None, sep=",").values
yy = pd.read_csv(divided_sets_path + "yy", header=None, sep=",")[0].values
Xt = pd.read_csv(divided_sets_path + "Xt", header=None, sep=",").values
yt = pd.read_csv(divided_sets_path + "yt", header=None, sep=",")[0].values
Tools.run_test(Xx, Xt, yy, yt, absolute_path)

