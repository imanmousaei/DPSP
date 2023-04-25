#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


import pandas as pd
import numpy as np
from numpy.random import seed
seed(1)

from models.mlp import MLP
from train.classification import Classification
from train.utils import *


def cross_validation(feature_matrix, label_matrix, event_num, seed, CV):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    index_all_class = get_index(label_matrix, event_num, seed, CV)
    matrix = []
    if type(feature_matrix) != list:
        matrix.append(feature_matrix)
        feature_matrix = matrix
        
    for k in range(CV):
        train_index = np.where(index_all_class != k)
        test_index = np.where(index_all_class == k)
        pred = np.zeros((len(test_index[0]), event_num), dtype=float)

        for i in range(len(feature_matrix)):
            x_train = feature_matrix[i][train_index]
            x_train = torch.Tensor(x_train)
            x_test = feature_matrix[i][test_index]
            x_test = torch.Tensor(x_test)
            
            # one-hot encoding
            y_train = label_matrix[train_index]
            y_train_one_hot = np.array(y_train)
            y_train_one_hot = (np.arange(y_train_one_hot.max() + 1)
                               == y_train[:, None]).astype(dtype='float32')
            y_train_one_hot = torch.Tensor(y_train_one_hot)
            
            # one-hot encoding
            y_test = label_matrix[test_index]
            y_test_one_hot = np.array(y_test)
            y_test_one_hot = (np.arange(y_test_one_hot.max() + 1)
                              == y_test[:, None]).astype(dtype='float32')
            y_test_one_hot = torch.Tensor(y_test_one_hot)


            model = MLP(input_size=new_feature.shape[1], output_size=event_num, droprate=droprate)
            classification = Classification(model, learning_rate=0.01, use_gpu=False)
            classification.train(batch_size=128, epochs=100, train_data=(x_train, y_train_one_hot), validation_data=(x_test, y_test_one_hot))
            
            pred += model(x_test)

        pred_score = pred / len(feature_matrix)
        pred_type = np.argmax(pred_score, axis=1)
        y_true = np.hstack((y_true, y_test))
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, pred_score))
    result_all, positive_negative, result_eve = evaluate(
        y_pred, y_score, y_true, event_num)
    return result_all, positive_negative, result_eve
    
    
def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_one_hot = label_binarize(y_test, np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, np.arange(event_num))

    precision, recall, th = multiclass_precision_recall_curve(
        y_one_hot, pred_score)

    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    result_all[5] = f1_score(y_test, pred_type, average='micro')
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = precision_score(y_test, pred_type, average='micro')
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[9] = recall_score(y_test, pred_type, average='micro')
    result_all[10] = recall_score(y_test, pred_type, average='macro')
    positive_negative = np.hstack(self_metric_calculate(y_test, pred_type))
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take(
            [i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                         average=None)
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary')
    return [result_all, positive_negative, result_eve]


if __name__ == '__main__':

    # Dataset1 (DS1):
    # event_num = 65
    # droprate = 0.3
    # vector_size = 572
    # df_drug = pd.read_pickle('DS1/df.pkl')
    # conn = sqlite3.connect("DS1/event.db")
    # feature_list = df_drug[["side", "target", "enzyme", "pathway", "smile"]]
    # extraction = pd.read_sql('select * from extraction;', conn)
    # mechanism = extraction['mechanism']
    # action = extraction['action']
    # drugA = extraction['drugA']
    # drugB = extraction['drugB']

    # Dataset2 (DS2):
    event_num = 100
    droprate = 0.3
    vector_size = 1258
    df_drug = pd.read_csv('DS2/drug_information_1258.csv')
    df_event = pd.read_csv('DS2/drug_interaction.csv')
    feature_list = df_drug[["target", "enzyme", "smile"]]
    mechanism = df_event['mechanism']
    action = df_event['action']
    drugA = df_event['drugA']
    drugB = df_event['drugB']

    seed = 0
    CV = 5
    new_feature, new_label = prepare(
        df_drug, feature_list, vector_size, mechanism, action, drugA, drugB)

    print(event_num)
    all_result, positive_negative, each_result = cross_validation(
        new_feature, new_label, event_num, seed, CV)
