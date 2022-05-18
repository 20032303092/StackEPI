import random
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, StandardScaler


def get_train_X_y(datasource, cell_name, feature_name):
    if datasource != "epivan" and datasource != "sept":
        raise ValueError("datasource must be 'epivan' or 'sept' !!!")
    trainPath = r'../../data/%s/%s/features/%s/%s_train.npz' % (datasource, cell_name, feature_name, cell_name)
    train_data = np.load(trainPath)  # ['X_en_tra', 'X_pr_tra', 'y_tra'] / ['X_en_tes', 'X_pr_tes', 'y_tes']
    X_en = train_data[train_data.files[0]]
    X_pr = train_data[train_data.files[1]]
    train_X = [np.hstack((item1, item2)) for item1, item2 in zip(X_en, X_pr)]
    # train_X = [np.hstack((item1, item2, item1 - item2, item1 * item2)) for item1, item2 in zip(X_en, X_pr)]
    # print(type(self.X))
    train_y = train_data[train_data.files[2]]
    return train_X, train_y


def get_test_X_y(datasource, cell_name, feature_name):
    testPath = r'../../data/%s/%s/features/%s/%s_test.npz' % (datasource, cell_name, feature_name, cell_name)
    test_data = np.load(testPath)  # ['X_en_tra', 'X_pr_tra', 'y_tra'] / ['X_en_tes', 'X_pr_tes', 'y_tes']
    X_en = test_data[test_data.files[0]]
    X_pr = test_data[test_data.files[1]]
    test_X = [np.hstack((item1, item2)) for item1, item2 in zip(X_en, X_pr)]
    # test_X = [np.hstack((item1, item2, item1 - item2, item1 * item2)) for item1, item2 in zip(X_en, X_pr)]
    test_X = np.array(test_X)
    # print(type(self.X))
    test_y = test_data[test_data.files[2]]
    return test_X, test_y


def get_data_np_dict(datasource, cell_name, feature_name, method_name):
    train_X, train_y = get_train_X_y(datasource, cell_name, feature_name)
    test_X, test_y = get_test_X_y(datasource, cell_name, feature_name)
    if method_name == "meta":
        pass
    else:
        print("experiment: %s %s_%s" % (cell_name, feature_name, method_name))
        print("trainSet len:[X=%s,y=%s]" % (len(train_X), len(train_y)))
        print("testSet len:[X=%s,y=%s]" % (len(test_X), len(test_y)))
        print(f"{feature_name} dim:", len(train_X[0]))

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    return {'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y}


def get_data_np_dict_train_val(datasource, cell_name, feature_name, method_name):
    data_X, data_y = get_train_X_y(datasource, cell_name, feature_name)
    # test_X, test_y = get_test_X_y(datasource, cell_name, feature_name)

    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.2, random_state=250,
                                                        stratify=data_y)
    if method_name == "meta":
        pass
    else:
        print("experiment: %s %s_%s" % (cell_name, feature_name, method_name))
        print("trainSet len:[X=%s,y=%s]" % (len(train_y), len(train_X)))
        print("testSet len:[X=%s,y=%s]" % (len(test_y), len(test_X)))
        print(f"{feature_name} dim:", len(train_X[0]))

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    return {'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y}


def shuffleData(X, y, seed=None):
    random.seed(seed)
    index = [i for i in range(len(X))]
    random.shuffle(index)
    # print(index)
    X = X[index]
    y = y[index]
    return X, y


def shuffle_data_list_dict(data_list_dict: dict, seed=None):
    train_X, train_y = shuffleData(data_list_dict["train_X"], data_list_dict["train_y"], seed)
    test_X, test_y = shuffleData(data_list_dict["test_X"], data_list_dict["test_y"], seed)
    return {'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y}


def scaler(data_list_dict: dict, ):
    train_X = scale(data_list_dict["train_X"])
    test_X = scale(data_list_dict["test_X"])
    train_y = data_list_dict["train_y"]
    test_y = data_list_dict["test_y"]
    print("data Scaler!")
    return {'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y}


def standardScaler(data_list_dict: dict):
    scaler = StandardScaler()
    scaler.fit(data_list_dict["train_X"])
    train_X = scaler.transform(data_list_dict["train_X"])
    test_X = scaler.transform(data_list_dict["test_X"])
    train_y = data_list_dict["train_y"]
    test_y = data_list_dict["test_y"]
    print("data standardScaler!")
    return {'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y}


def feature_select_RFE(data_list_dict: dict, estimator):
    # rfe = RFE(estimator=lightgbm.sklearn.LGBMClassifier(device='gpu'), step=1)
    rfe = RFE(estimator=estimator, step=1)
    rfe.fit(data_list_dict["train_X"], data_list_dict["train_y"])
    print("RFE n_features:", rfe.n_features_)
    data_list_dict.update({"train_X": rfe.transform(data_list_dict["train_X"])})
    data_list_dict.update({"test_X": rfe.transform(data_list_dict["test_X"])})
    return data_list_dict


if __name__ == '__main__':
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3, ], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8]])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    X, y = shuffleData(X, y, seed=1)
    print(X, y)
