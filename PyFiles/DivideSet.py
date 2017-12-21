import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sys import argv

def prepare_data(path, name_and_position_of_file, is_header_present, name_or_number_of_target_column,
                 separator, percent_of_test_examples, is_oversampling_enabled):
    save_path = path + "/DividedSets/"
    if is_header_present:
        df = pd.read_csv(name_and_position_of_file, sep=separator)
        y = df[name_or_number_of_target_column].values
        df = df.drop(name_or_number_of_target_column, axis=1)
    else:
        df = pd.read_csv(name_and_position_of_file, header=None, sep=separator)
        y_classification = df.columns[int(name_or_number_of_target_column) - 1]
        y = df[y_classification].values
        df = df.drop(y_classification, axis=1)
    df = df.fillna(value=df.mean())
    df_norm = (df - df.min()) / (df.max() - df.min())
    X = df_norm.values
    Xx, Xt, yy, yt = train_test_split(X, y, test_size=percent_of_test_examples, stratify=y)
    if is_oversampling_enabled:
        smt = SMOTE()
        Xx, yy = smt.fit_sample(Xx, yy)
        Xx, yy = shuffle(Xx, yy)
    pd.DataFrame(Xx).to_csv(save_path + "Xx", index=False, header=False)
    pd.DataFrame(yy).to_csv(save_path + "yy", index=False, header=False)
    pd.DataFrame(Xt).to_csv(save_path + "Xt", index=False, header=False)
    pd.DataFrame(yt).to_csv(save_path + "yt", index=False, header=False)

if __name__ == "__main__":
    name_of_file = argv[1]
    absolute_path = argv[2]
    is_header_present = argv[3]
    name_or_number_of_target_column = argv[4]
    separator = argv[5]
    percent_of_test_examples = argv[6]
    is_oversampling_enabled = argv[7]
    if is_header_present == 'true':
        is_header_present = True
    else:
        is_header_present = False
    if is_oversampling_enabled == 'true':
        is_oversampling_enabled = True
    else:
        is_oversampling_enabled = False
    prepare_data(absolute_path, absolute_path + "/Data/" + name_of_file, is_header_present, name_or_number_of_target_column,
                    separator, float(percent_of_test_examples), is_oversampling_enabled)