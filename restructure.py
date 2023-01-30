import pandas as pd

train_dataframe = pd.read_csv("./data/CONDA_valid.csv",dtype=str,encoding='utf-8')
train_dataframe = train_dataframe[["utterance","intentClass"]]
# test_dataframe = pd.read_csv("./data/CONDA_valid.csv",dtype=str,encoding='utf-8')
# test_dataframe = train_dataframe[["utterance","intentClass"]]

train_dataframe.dropna(inplace=True)
# test_dataframe.dropna(inplace=True)

binary_classes = []
for index, row in train_dataframe.iterrows():
    binary_classes.append( 1 if  row["intentClass"] == "E" or row["intentClass"] == "I" else 0)   

train_dataframe["binary"] = binary_classes
print(train_dataframe.head())

directory = 'text_files/valid/nontoxic'

for index, row in train_dataframe.iterrows():

    if (row["binary"] == 0):
        with open(directory + "/" + str(index) + '.txt', 'w',encoding="utf-8") as file:
            file.write(row["utterance"])

directory = 'text_files/valid/toxic'

for index, row in train_dataframe.iterrows():

    if (row["binary"] == 1):
        with open(directory + "/" + str(index) + '.txt', 'w',encoding="utf-8") as file:
            file.write(row["utterance"])
