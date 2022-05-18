import copy
import csv
import math
import os
import time
from inspect import signature
from itertools import product

import lightgbm
import deepforest
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import precision_score, confusion_matrix, recall_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from ML.EPIconst import EPIconst
from utils.utils_data import get_data_np_dict


def _get_score_dict(scoring):
    score_dict = {}
    if isinstance(scoring, str):
        score_dict.update({scoring + '_score': scoring})
    else:
        for item in scoring:
            score_dict.update({item + '_score': item})
    # score_dict = dict(sorted(score_dict.items(), key=lambda x: x[0], reverse=False))
    # print(score_dict)
    return score_dict


def get_scoring_result(scoring, y, y_pred, y_prob, y_score=None, is_base_score=True):
    process_msg = ""
    if y_score is None:
        y_score = y_prob
    module_name = __import__("sklearn.metrics", fromlist='*')
    # print('\n'.join(['%s:%s' % item for item in module_name.__dict__.items()]))
    score_dict = _get_score_dict(scoring)
    # print(score_dict)
    # start get_scoring_result
    score_result_dict = {}
    if is_base_score:
        TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        score_result_dict.update({"Total": len(y), "TP": TP, "TN": TN, "FP": FP, "FN": FN, "precision": precision,
                                  "recall": recall})
        process_msg += "total=%s, TP=%s, TN=%s, FP=%s, FN=%s; precision=%.3f, recall=%.3f\n" \
                       % (len(y), TP, TN, FP, FN, precision, recall)
    for k, v in score_dict.items():
        # print("===", k)
        score_func = getattr(module_name, k)
        sig = signature(score_func)
        # print(sig)
        y_flag = str(list(sig.parameters.keys())[1])
        # print(y_flag)
        if y_flag == 'y_pred':
            y_flag = y_pred
        elif y_flag == 'y_prob':
            y_flag = y_prob
        elif y_flag == 'y_score':
            y_flag = y_score
        else:
            raise ValueError("having new metrics that its 2nd param is not y_pred y_prob or y_score in sklearn !!!")
        if y_flag is None:
            raise ValueError(k, "%s is None !!!" % (y_flag))
        score_result = score_func(y, y_flag)
        # accuracy: (test=0.926)
        # print("%s: (test=%s)" % (v, score_result), end=" ")
        process_msg += "%s: (test=%.3f) " % (v, score_result)
        # print("%s: (test=%.3f) ===" % (v, score_result))
        score_result_dict.update({v: score_result})
    # print("score_result_dict:", score_result_dict)
    return process_msg, score_result_dict


def writeCVRank2csv(met_grid, clf, ex_dir_path, cell_name, computer, index=None):
    print("write rank test to csv!!!")
    csv_rows_list = []
    header = []
    csv_rows_list.append(clf.cv_results_['params'])
    header.append('params')

    for m in met_grid:
        header.append(m)
        csv_rows_list.append(clf.cv_results_[m])

    for m in met_grid:
        rank_test_score = 'rank_test_' + m
        mean_test_score = 'mean_test_' + m
        std_test_score = 'std_test_' + m
        header.append(rank_test_score)
        header.append(mean_test_score)
        header.append(std_test_score)
        csv_rows_list.append(clf.cv_results_[rank_test_score])
        csv_rows_list.append(clf.cv_results_[mean_test_score])
        csv_rows_list.append(clf.cv_results_[std_test_score])

    results = list(zip(*csv_rows_list))

    ex_rank_dir_path = r'%s/rank' % ex_dir_path
    if not os.path.exists(ex_dir_path):
        os.mkdir(ex_dir_path)
        print(ex_dir_path, "created !!!")

    if not os.path.exists(ex_rank_dir_path):
        os.mkdir(ex_rank_dir_path)
        print(ex_rank_dir_path, "created !!!")

    feature_method_ensembleStep = ex_rank_dir_path.split('/')[-1]

    file_name = r'%s/%s_%s_rank_%s_%s.csv' % (ex_rank_dir_path, cell_name, feature_method_ensembleStep, index, computer)
    if index is None:
        file_name = r'%s/%s_%s_rank_%s.csv' % (ex_rank_dir_path, cell_name, feature_method_ensembleStep, computer)

    with open(file_name, 'wt', newline='')as f:
        f_csv = csv.writer(f, delimiter=",")
        f_csv.writerow(header)
        f_csv.writerows(results)
        f.close()
    print(file_name, "write over!!!")


def writeRank2csv(met_grid, clf, ex_dir_path, cell_name, computer, index=None):
    print("write rank test to csv!!!")
    csv_rows_list = []
    header = []
    csv_rows_list.append(clf.cv_results_['params'])
    header.append('params')

    for m in met_grid:
        header.append(m)
        csv_rows_list.append(clf.cv_results_[m])

    for m in met_grid:
        rank_test_score = 'rank_test_' + m
        header.append(rank_test_score)
        csv_rows_list.append(clf.cv_results_[rank_test_score])

    results = list(zip(*csv_rows_list))

    ex_rank_dir_path = r'%s/rank' % ex_dir_path
    if not os.path.exists(ex_dir_path):
        os.mkdir(ex_dir_path)
        print(ex_dir_path, "created !!!")

    if not os.path.exists(ex_rank_dir_path):
        os.mkdir(ex_rank_dir_path)
        print(ex_rank_dir_path, "created !!!")

    feature_method_ensembleStep = ex_dir_path.split('/')[-1]

    file_name = r'%s/%s_%s_rank_%s_%s.csv' % (ex_rank_dir_path, cell_name, feature_method_ensembleStep, index, computer)
    if index is None:
        file_name = r'%s/%s_%s_rank_%s.csv' % (ex_rank_dir_path, cell_name, feature_method_ensembleStep, computer)

    with open(file_name, 'wt', newline='')as f:
        f_csv = csv.writer(f, delimiter=",")
        f_csv.writerow(header)
        f_csv.writerows(results)
        f.close()
    print(file_name, "write over!!!")


def time_since(start):
    s = time.time() - start
    # s = 62 - start
    if s < 60:
        return '%.2fs' % s
    elif 60 < s and s < 3600:
        s = s / 60
        return '%.2fmin' % s
    else:
        m = math.floor(s / 60)
        s -= m * 60
        h = math.floor(m / 60)
        m -= h * 60
        return '%dh %dm %ds' % (h, m, s)


class RunAndScore:

    def __init__(self, data_list_dict, estimator, parameters, scoring, refit, n_jobs, verbose=0):
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.estimator = estimator
        self.scoring = sorted(scoring)
        self.refit = refit
        self.data_list_dict = data_list_dict
        self.candidate_params = self.get_candidate_params(parameters)

        self.all_out = self.run_and_score()
        self.cv_results_ = self.get_cv_results()
        self.best_estimator_params_idx_, self.best_estimator_params_ = self.get_best_estimator_params()
        self.best_estimator_ = self.get_best_estimator()
        self.best_scoring_result = self.get_best_scoring_result()

    def model_fit(self, estimator, train_X, train_y):
        """
        train model
        :param estimator:
        :param train_X:
        :param train_y:
        :return:
        """
        estimator.fit(train_X, train_y)

    def model_predict(self, model, test_X):
        """
        get y_pred
        :param model:
        :return:
        """
        y_pred = model.predict(test_X)
        return y_pred

    def model_predict_proba(self, model, test_X):
        """
        get y_prob_temp
        :param model:
        :return:
        """
        y_prob_temp = model.predict_proba(test_X)
        return y_prob_temp

    def set_estimator_params(self, estimator, params: dict):
        """
        set estimator parameters
        :param estimator:
        :param params:
        :return:
        """
        if self.verbose > 0:
            print("set params:", params)
        if isinstance(estimator, lightgbm.sklearn.LGBMClassifier):
            estimator.set_params(**params)
        else:
            for k, v in params.items():
                setattr(estimator, k, v)
        if self.verbose > 0:
            if isinstance(estimator, lightgbm.sklearn.LGBMClassifier):
                print(estimator.get_params())
            else:
                print(estimator.__dict__)
        # if isinstance(estimator, deepforest.CascadeForestClassifier):
        #     lgb = lightgbm.sklearn.LGBMClassifier()
        #     lgb.set_params(**EPIconst.ModelInitParams.lightgbm)
        #     lgb.set_params(**EPIconst.BaseModelParams.GM12878_eiip_lightgbm)
        #     lgb.set_params(**{"n_jobs": 10})
        #     cascade_estimators = [lgb for i in range(params["n_estimators"])]
        #     estimator.set_estimator(cascade_estimators)
        return estimator

    def get_candidate_params(self, parameters):
        """
        return params_dict_list
        :param parameters:
        :return:
        """
        return list(ParameterGrid(parameters))

    def fit_and_predict(self, cand_idx, params):
        # print("==fit and predict==")
        n_candidates = len(self.candidate_params)
        process_msg = "[fit {}/{}] END ".format(cand_idx, n_candidates)
        start = time.time()
        deep_forest = copy.deepcopy(self.estimator)
        model = self.set_estimator_params(deep_forest, params)
        self.model_fit(model, self.data_list_dict["train_X"], self.data_list_dict["train_y"])
        # print("==fit over,start predicting==")
        y_pred = self.model_predict(model, self.data_list_dict["test_X"])
        # print("==predicted,start predict_proba==")
        y_pred_prob_temp = self.model_predict_proba(model, self.data_list_dict["test_X"])
        # print("==predicted_proba==")
        y_pred_prob = []
        if (y_pred[0] == 1 and y_pred_prob_temp[0][0] >= 0.5) or (y_pred[0] == 0 and y_pred_prob_temp[0][0] < 0.5):
            y_pred_prob = y_pred_prob_temp[:, 0]
        else:
            y_pred_prob = y_pred_prob_temp[:, 1]

        params_msg = ""
        for k, v in params.items():
            params_msg += "{}={}, ".format(k, v)

        process_msg += params_msg[: -2] + '; '
        # print("==get_scoring_result==")
        [score_result_msg, score_result_dict] = get_scoring_result(self.scoring, self.data_list_dict["test_y"], y_pred,
                                                                   y_pred_prob)
        process_msg += score_result_msg
        process_msg += time_since(start)
        # print("==getted_scoring_result==")
        print(process_msg)
        # print([params, score_result_dict])
        return [params, score_result_dict]

    def run_and_score(self):
        """
        all_out = [out0,out1,out2,...]
        out = [[params, score_result_dict],[params, score_result_dict],...]
        score_result_dict = {"score0":s0,"score1":s1,...}
        :return:
        """
        n_candidates = len(self.candidate_params)
        print("Fitting, totalling {0} fits".format(n_candidates))
        # if self.n_jobs > 1:
        parallel = Parallel(n_jobs=self.n_jobs)
        with parallel:
            all_out = []
            out = parallel(delayed(self.fit_and_predict)
                           (cand_idx, params)
                           for cand_idx, params in enumerate(self.candidate_params, 1))
            # print(out)
            n_candidates = len(self.candidate_params)
            if len(out) < 1:
                raise ValueError('No fits were performed. '
                                 'Was the CV iterator empty? '
                                 'Were there no candidates?')
            elif len(out) != n_candidates:
                raise ValueError('cv.split and cv.get_n_splits returned '
                                 'inconsistent results. Expected {} '
                                 'splits, got {}'
                                 .format(n_candidates, len(out)))
            # print("score_result_dict:", out)
            all_out.extend(out)

        return all_out

    def _get_score_dict(self):
        """
        score_dict: {"xxx_score":xxx,...}
                    {"score_method_name":"score_name"}
        :return:
        """
        score_dict = {}
        if isinstance(self.scoring, str):
            score_dict.update({self.scoring + '_score': self.scoring})
        else:
            for item in self.scoring:
                score_dict.update({item + '_score': item})
        score_dict = dict(sorted(score_dict.items(), key=lambda x: x[0], reverse=False))
        # print(score_dict)
        return score_dict

    def get_cv_results(self):
        """
        cv_results = {"params":[{sample1_params},{sample2_params}],
                      "rank_test_xx1":[],"rank_test_xx2":[],}
        :return:
        """
        cv_results = {}
        cv_results.update({"params": []})

        score_dict = self._get_score_dict()
        for k, score_name in score_dict.items():
            cv_results.update({"rank_test_%s" % score_name: []})
            cv_results.update({score_name: np.array([])})

        # set data into cv_results by column
        for cv_out in self.all_out:
            cv_results["params"].append(cv_out[0])
            cv_out_score = cv_out[1]
            for k, score_name in score_dict.items():
                cv_results[score_name] = np.append(cv_results[score_name], cv_out_score[score_name])
                # cv_results[v].append(cv_out_score[v])
        # print("not rank:", cv_results)
        cv_results = self._rank_cv_result(cv_results)
        # print("ranked:", cv_results)
        return cv_results

    def _rank_cv_result(self, cv_results):
        for item in self.scoring:
            # sorted by mean
            obj = pd.Series(cv_results[item])
            c = obj.rank(ascending=False, method="min")
            # print(c.values.astype(int))
            cv_results.update({"rank_test_%s" % item: c.values.astype(int)})

            # # sorted by mean and std
            # df = pd.DataFrame({"mean": cv_results["mean_test_%s" % item], "std": cv_results["std_test_%s" % item]})
            # cv_results["rank_test_%s" % item] = df.sort_values(by=['std', 'mean'])['mean'] \
            #     .rank(method='first', ascending=False).values.astype(int)
        return cv_results

    def get_best_estimator_params(self):
        if isinstance(self.refit, str):
            idx = list(self.cv_results_["rank_test_%s" % self.refit]).index(1)
        return idx, self.cv_results_["params"][idx]

    def get_best_estimator(self):
        estimator = copy.deepcopy(self.estimator)
        estimator = self.set_estimator_params(estimator, self.best_estimator_params_)
        self.model_fit(estimator, self.data_list_dict["train_X"], self.data_list_dict["train_y"])
        return estimator

    def get_best_scoring_result(self):
        process_msg = "["
        result = self.all_out[self.best_estimator_params_idx_]

        for k, v in result[1].items():
            if isinstance(v, int):
                process_msg += "%s: (test=%d) " % (k, v)
            else:
                process_msg += "%s: (test=%.3f) " % (k, v)
        return process_msg + "]"
