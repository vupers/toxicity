import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report,f1_score

def get_dataframe(name = 'rewire'):
    """ 
    Returns df if given folder exists, None otherwise.
    """
    try:
        scoreDataframe = pd.read_csv("results/" + name + ".csv")
        scoreDataframe = scoreDataframe[["utterance","intentClass","score"]]
        return scoreDataframe
    except:
        return None


def compute_classifications(df, threshold):
    classifications = []
    for index, row in df.iterrows():
        classifications.append(1 if  row["score"] >= threshold else 0)
    return classifications

def generate_continuous_plot(model_name, score_dataframe,thresholds):
    accuracies = []
    overall = []
    for threshold in thresholds:
        score_dataframe["classifications"] = compute_classifications(score_dataframe,threshold)
        overall_accuracy = accuracy_score(score_dataframe['binary_classes'], score_dataframe["classifications"])
        overall.append(overall_accuracy)
        matrix = confusion_matrix(score_dataframe['binary_classes'], score_dataframe["classifications"])
        individual_accuracies = matrix.diagonal()/matrix.sum(axis=1)
        accuracies.append(individual_accuracies)

    plt.plot(thresholds,accuracies,'o-',label=["Non-Toxic Messages Only","Toxic Messages Only"])
    plt.plot(thresholds,overall,'--',label = "Combined Accuracy")
    plt.xlabel("Toxicity threshold")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name}: Variation in accuracy @varying toxicity thresholds (0.05 range)")
    plt.legend()
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.grid()
    plt.savefig(f"analysis/{model_name}/continuous_plot.png")
    plt.close()

def generate_confusion_matrices(model_name,score_dataframe):
    thresholds = [0.2,0.5,0.7,0.9]
    class_names = ["Non-toxic", "Toxic"]
    matrices = []
    percentages = []
    labels_list = []
    for threshold in thresholds:
        score_dataframe["classifications"] = compute_classifications(score_dataframe,threshold)
        matrix = confusion_matrix(score_dataframe['binary_classes'], score_dataframe["classifications"])
        matrices.append(matrix)
        group_counts = [f'{value:d}' for value in
                        matrix.flatten()]
        group_percentages = ['{0:.2%}'.format(value) for value in
                            matrix.flatten()/np.sum(matrix)]
        labels = [f'{v1}\n{v2}' for v1, v2 in
                zip(group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        labels_list.append(labels)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,figsize=(11,9))
    fig.suptitle(f"{model_name}: Confusion Matrices @changing thresholds", fontsize=17)
    sns.heatmap(matrices[0], annot=labels_list[0], ax=ax1, xticklabels=class_names, yticklabels=class_names, cmap='Blues', fmt = '')
    sns.heatmap(matrices[1], annot=labels_list[1], ax=ax2, xticklabels=class_names, yticklabels=class_names, cmap='Blues',fmt = '')
    sns.heatmap(matrices[2], annot=labels_list[2], ax=ax3, xticklabels=class_names, yticklabels=class_names, cmap='Blues',fmt = '')
    sns.heatmap(matrices[3], annot=labels_list[3], ax=ax4, xticklabels=class_names, yticklabels=class_names, cmap='Blues',fmt = '')
    ax1.set_title(f"Threshold = {thresholds[0]}")
    ax2.set_title(f"Threshold = {thresholds[1]}")
    ax3.set_title(f"Threshold = {thresholds[2]}")
    ax4.set_title(f"Threshold = {thresholds[3]}")
    # plt.show()
    plt.savefig(f"analysis/{model_name}/matrices.png") 
    plt.close()
    
def binarize(score_dataframe):
    binary_classes = []
    for index, row in score_dataframe.iterrows():
        binary_classes.append( 1 if  row["intentClass"] == "E" or row["intentClass"] == "I" else 0)   
    return binary_classes

model_name = 'bert'  #insert 'perspective', 'rewire' or 'detoxify' 
score_dataframe = get_dataframe(model_name)
score_dataframe['binary_classes'] = binarize(score_dataframe)
generate_confusion_matrices(model_name,score_dataframe)
thresholds = np.arange(0.05, 1.00, 0.05)
generate_continuous_plot(model_name, score_dataframe,thresholds)


