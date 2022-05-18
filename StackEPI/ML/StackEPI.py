import os
import sys
import time
import warnings
from itertools import product

import deepforest
import joblib
import lightgbm.sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import thundersvm
import xgboost

warnings.filterwarnings("ignore")
root_path = os.path.abspath(os.path.dirname(__file__)).split('stackEPI')
sys.path.extend([root_path[0] + 'stackEPI'])

from deepforest import CascadeForestClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from thundersvm import SVC
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from utils.utils_data import get_data_np_dict
from ML.ml_def import time_since, get_scoring_result
from ML.EPIconst import EPIconst


def get_new_feature(cell_name, selected_feature_names, selected_method_names, save_path):
    test_y_pred = []
    test_y_proba = []
    train_y_pred = []
    train_y_proba = []
    if isinstance(selected_method_names, str):
        selected_method_names = [selected_method_names]
    if isinstance(selected_feature_names, str):
        selected_feature_names = [selected_feature_names]
    data_value = {}
    for item in product(selected_feature_names, selected_method_names):
        ex_item = cell_name + "_" + "_".join(item)
        feature_name = item[0]
        method_name = item[1]
        if ex_item.__contains__("HeLa-S3"):
            ex_item = "HeLa_S3" + "_" + feature_name + "_" + method_name
        model_params = getattr(EPIconst.BaseModelParams, ex_item)
        init_params = getattr(EPIconst.ModelInitParams, method_name)
        estimator = estimators[method_name]()
        estimator.set_params(**init_params)
        estimator.set_params(**model_params)
        print(ex_item, ":", estimator)
        data_value = get_data_np_dict(datasource, cell_name, feature_name, EPIconst.MethodName.ensemble)

        start_time = time.time()
        estimator.fit(data_value["train_X"], data_value["train_y"])

        # """
        # save model
        # """
        # model_save(estimator, ex_item + "_base", save_path)

        """
        get new testSet
        """
        y_pred = estimator.predict(data_value["test_X"])
        y_prob_temp = estimator.predict_proba(data_value["test_X"])

        if (y_pred[0] == 1 and y_prob_temp[0][0] >= 0.5) or (y_pred[0] == 0 and y_prob_temp[0][0] < 0.5):
            y_proba = y_prob_temp[:, 0]
        else:
            y_proba = y_prob_temp[:, 1]

        test_y_pred.append(y_pred)
        test_y_proba.append(y_proba)
        # get base estimator score
        scoring = sorted(['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy'])
        process_msg, score_result_dict = get_scoring_result(scoring, data_value["test_y"], y_pred, y_proba)
        # print(ex_item, ":", process_msg, "\n")
        print("{0}:{1} \nbase_estimator_train_time:{2}".format(ex_item, process_msg, time_since(start_time)))

        # get new trainSet
        y_pred = estimator.predict(data_value["train_X"])
        y_prob_temp = estimator.predict_proba(data_value["train_X"])
        if (y_pred[0] == 1 and y_prob_temp[0][0] >= 0.5) or (y_pred[0] == 0 and y_prob_temp[0][0] < 0.5):
            y_proba = y_prob_temp[:, 0]
        else:
            y_proba = y_prob_temp[:, 1]
        # print(y_proba)
        train_y_pred.append(y_pred)
        train_y_proba.append(y_proba)

    test_y_pred_np = np.array(test_y_pred)
    test_y_prob_np = np.array(test_y_proba)
    train_y_pred_np = np.array(train_y_pred)
    train_y_prob_np = np.array(train_y_proba)

    train_y = data_value["train_y"]
    print(train_y)
    test_X_pred = test_y_pred_np.T
    test_X_prob = test_y_prob_np.T

    test_y = data_value["test_y"]
    train_X_pred = train_y_pred_np.T
    train_X_prob = train_y_prob_np.T

    return {'train_X': {"train_X_pred": train_X_pred, "train_X_prob": train_X_prob}, 'train_y': train_y,
            'test_X': {"test_X_pred": test_X_pred, "test_X_prob": test_X_prob}, 'test_y': test_y}


def get_ensemble_data(new_feature, datatype):
    data_list_dict = {}
    if datatype == "pred":
        print("use pred feature !!!")
        data_list_dict = {'train_X': new_feature['train_X']["train_X_pred"], 'train_y': new_feature['train_y'],
                          'test_X': new_feature['test_X']["test_X_pred"], 'test_y': new_feature['test_y']}
    elif datatype == "prob":
        print("use prob feature !!!")
        data_list_dict = {'train_X': new_feature['train_X']["train_X_prob"], 'train_y': new_feature['train_y'],
                          'test_X': new_feature['test_X']["test_X_prob"], 'test_y': new_feature['test_y']}
    elif datatype == "prob-pred":
        print("use prob_pred feature !!!")
        train_X_prob = new_feature["train_X"]["train_X_prob"]
        test_X_prob = new_feature["test_X"]["test_X_prob"]
        train_X_pred = new_feature["train_X"]["train_X_pred"]
        test_X_pred = new_feature["test_X"]["test_X_pred"]
        train_X = np.hstack((train_X_prob, train_X_pred))
        test_X = np.hstack((test_X_prob, test_X_pred))
        train_y = new_feature["train_y"]
        test_y = new_feature["test_y"]
        data_list_dict = {'train_X': train_X, 'train_y': train_y,
                          'test_X': test_X, 'test_y': test_y}
    return data_list_dict


def get_meta_result(cell_name, ensemble_consist_type, ensemble_feature_name, meta_method_name, ensemble_data,
                    save_path):
    """
    set meta estimator
    """
    # ex_item: GM12878_6f5m_prob_mlp
    ex_item = cell_name + "_" + ensemble_consist_type + "_" + ensemble_feature_name + "_" + meta_method_name
    if ex_item.__contains__("HeLa-S3"):
        ex_item = "HeLa_S3" + "_" + ensemble_consist_type + "_" + ensemble_feature_name + "_" + meta_method_name
    meta_model_params = getattr(EPIconst.MetaModelParams, ex_item)
    init_params = getattr(EPIconst.ModelInitParams, meta_method_name)
    meta_estimator = estimators[meta_method_name]()
    meta_estimator.set_params(**init_params)
    meta_estimator.set_params(**meta_model_params)
    print("%s:%s" % (ex_item, meta_estimator))

    """
    fit and prediction
    """
    start_time = time.time()
    meta_estimator.fit(ensemble_data["train_X"], ensemble_data["train_y"])
    train_time = time_since(start_time)
    # """
    # save model
    # """
    # model_save(meta_estimator, ex_item + "_meta", save_path)

    y_pred = meta_estimator.predict(ensemble_data["test_X"])
    y_prob_temp = meta_estimator.predict_proba(ensemble_data["test_X"])
    if (y_pred[0] == 1 and y_prob_temp[0][0] >= 0.5) or (y_pred[0] == 0 and y_prob_temp[0][0] < 0.5):
        y_proba = y_prob_temp[:, 0]
    else:
        y_proba = y_prob_temp[:, 1]
    # print(y_proba)

    """
    get meta_estimator score
    """
    scoring = sorted(['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy'])
    process_msg, score_result_dict = get_scoring_result(scoring, ensemble_data["test_y"], y_pred, y_proba)

    print("{0}:{1} meta_train_time:{2}\n".format(ex_item, process_msg, train_time))
    return train_time, score_result_dict


def stackEPI_result(cell_name, save_path):
    ensemble_data_time_start = time.time()
    new_feature = get_new_feature(cell_name, EPIconst.FeatureName.all,
                                  EPIconst.MethodName.all, save_path)
    ensemble_data = get_ensemble_data(new_feature, ensemble_feature_name)
    ensemble_data_time = time_since(ensemble_data_time_start)
    print("ensemble_data_time:", ensemble_data_time, "\n")
    """
    get_meta_result
    """
    meta_method_name_list = ["mlp", "logistic", "deepforest", "lightgbm", "rf", "svm", "xgboost"]
    for meta_method_name in meta_method_name_list:
        train_time, score_result_dict = get_meta_result(cell_name, ensemble_consist_type, ensemble_feature_name,
                                                        meta_method_name, ensemble_data, save_path)
        # print("train_time:", train_time)
        # print("score_result_dict:", score_result_dict)


def model_save(estimator, save_name, save_path="./model"):
    save_path = save_path + "/" + save_name
    if isinstance(estimator, deepforest.CascadeForestClassifier):
        print("save deepforest")
        estimator.save(save_path)
    elif isinstance(estimator, lightgbm.sklearn.LGBMClassifier):
        print("save lightgbm")
        joblib.dump(estimator, save_path + ".pkl")
    elif isinstance(estimator, sklearn.ensemble.RandomForestClassifier):
        print("save randomforest")
        joblib.dump(estimator, save_path + ".pkl")
    elif isinstance(estimator, thundersvm.SVC):
        print("save svm")
        estimator.save_to_file(save_path + ".pkl")
    elif isinstance(estimator, xgboost.XGBClassifier):
        print("save xgboost")
        estimator.save_model(save_path + ".json")
    elif isinstance(estimator, sklearn.linear_model.LogisticRegression):
        print("save logistic")
        joblib.dump(estimator, save_path + ".pkl")
    elif isinstance(estimator, sklearn.neural_network.MLPClassifier):
        print("save mlp")
        joblib.dump(estimator, save_path + ".pkl")
    else:
        raise Exception("estimator model saving error!")


"""
cell and feature choose
"""
estimators = {"xgboost": XGBClassifier, "svm": SVC, "rf": RandomForestClassifier,
              "deepforest": CascadeForestClassifier, "mlp": MLPClassifier, "logistic": LogisticRegression,
              "lightgbm": LGBMClassifier}
datasource = 'epivan'
ensemble_feature_names = ['prob', 'pred', 'prob-pred']
ensemble_feature_name = ensemble_feature_names[0]
ensemble_consist_types = ["6f5m", "4f2m"]
ensemble_consist_type = ensemble_consist_types[1]
ensemble_steps = ["base", "meta"]
ensemble_step = ensemble_steps[1]
computers = ["2080ti", "3070", "3090", "p400"]
computer = computers[1]
is_StandardScaler = False
"""
dataset Setting
"""
# EPIconst.MethodName.all.remove("lightgbm")
# EPIconst.MethodName.all.remove("xgboost")
# EPIconst.FeatureName.all.remove("dpcp")
# EPIconst.FeatureName.all.remove("tpcp")
# # EPIconst.FeatureName.all.remove("cksnap")
# # EPIconst.FeatureName.all.remove("kmer")
# # EPIconst.FeatureName.all.remove("pseknc")
# # EPIconst.FeatureName.all.remove("eiip")
# EPIconst.MethodName.all.remove("deepforest")
# EPIconst.MethodName.all.remove("svm")
# EPIconst.MethodName.all.remove("rf")
# # EPIconst.MethodName.all.remove("lightgbm")
# # EPIconst.MethodName.all.remove("xgboost")
if ensemble_consist_type == "4f2m":
    EPIconst.FeatureName.all.remove("dpcp")
    EPIconst.FeatureName.all.remove("tpcp")
    # EPIconst.FeatureName.all.remove("cksnap")
    # EPIconst.FeatureName.all.remove("kmer")
    # EPIconst.FeatureName.all.remove("pseknc")
    # EPIconst.FeatureName.all.remove("eiip")
    EPIconst.MethodName.all.remove("deepforest")
    EPIconst.MethodName.all.remove("svm")
    EPIconst.MethodName.all.remove("rf")

names = ["GM12878", "HeLa-S3", "HUVEC", "IMR90", "K562", "NHEK"]

save_path = "./model/model"
for cell_name in names:
    print("stackEPI_result", "#" * 80, ">>>>>>>>>>>>%s" % cell_name)
    stackEPI_result(cell_name, save_path)
