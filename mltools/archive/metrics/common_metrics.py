import numpy as np
import pandas as pd

def get_accuracy(confusion_matrix):
    return np.sum(np.diag(confusion_matrix)) / (np.sum(confusion_matrix) + EPSILON)

def get_actual_data_count(confusion_matrix):
    return np.sum(confusion_matrix, axis=1)

def get_precision(confusion_matrix):
    return np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=0) + EPSILON)

def get_recall(confusion_matrix):
    return np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + EPSILON)

def get_fvalue(confusion_matrix):
    precision = get_precision(confusion_matrix)
    recall = get_recall(confusion_matrix)
    return 2 * precision * recall / (precision + recall + EPSILON)

def get_confusion_matrix(actual_y, pred_y):
    confusion_matrix = np.zeros((2, 2))
    for i, j in zip(actual_y, pred_y):
        confusion_matrix[i, j] += 1

    return confusion_matrix

def get_confusion_dataframe(confusion_matrix, label_dict):
    preds = ['Pred_{}'.format(label_dict[i]) for i in range(len(label_dict))]
    actuals = ['Actual_{}'.format(label_dict[i]) for i in range(len(label_dict))]

    return pd.DataFrame(confusion_matrix, columns=preds, index=actuals)

def get_measure_dataframe(confusion_matrix, label_converter, ascending=True, row_count=None):
    measure_dataframe = pd.DataFrame(
        {
            'data_count': get_actual_data_count(confusion_matrix),
            'precision': get_precision(confusion_matrix),
            'recall': get_recall(confusion_matrix),
            'F-Value': get_fvalue(confusion_matrix)
        },
        index=[label_converter.detokenize(i) for i in range(label_converter.label_count)]
    )

    measure_dataframe = measure_dataframe.sort_values(by=['data_count'], ascending=ascending)
    if row_count is not None:
        measure_dataframe = measure_dataframe.iloc[:row_count]

    return measure_dataframe
