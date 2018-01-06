import pandas as pd
from PyFiles import Tools
from sys import argv

number_of_iterations = int(argv[1])
absolute_path = argv[2]
path = absolute_path + "/PyResults/"
path_to_save_files = absolute_path + "/PyProcessedResults/"
names = ["train_acc", "test_acc", "train_scores", "test_scores", "train_RMSE", "test_RMSE"]
files = [path + name for name in names]
files_to_save = [path_to_save_files + name for name in names]
statistics = [pd.read_csv(file, header=None)[0].values for file in files]
for i, stat in enumerate(statistics):
    batch_size = len(stat)//number_of_iterations
    divided_stat = [stat[j*batch_size:(j+1)*batch_size] for j in range(number_of_iterations)]
    mean_output = Tools.find_avg_of_vectors_by_column(divided_stat)
    pd.DataFrame(mean_output).to_csv(files_to_save[i], index=False, header=False)