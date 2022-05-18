import os
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model import get_model
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

index = 0
names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
header = ["cell line"]
for i in range(0, 15, 1):
    header.append("epoch-" + str(i))
rows = []
for name in names:
    row = [name]
    for i in range(0, 15, 1):
        model = get_model()
        # model.load_weights("./model_back/specificModel/%sModel.h5" % m)
        model.load_weights("./model/specificModel_%s/%sModel%s.h5" % (index, name, i))
        Data_dir = '../data/epivan/%s/' % name
        test = np.load(Data_dir + '%s_test.npz' % name)
        X_en_tes, X_pr_tes, y_tes = test['X_en_tes'], test['X_pr_tes'], test['y_tes']
        # print("y_test:", y_tes.shape)
        # print("y_test:", y_tes)
        """
        y_test: (2625,)
        y_test: [1. 1. 1. ... 0. 0. 0.]
        """
        print(
            "****************Testing %s cell line specific model_back on %s cell line****************" % (name, name))
        y_pred = model.predict([X_en_tes, X_pr_tes])  # y_pred: (2625, 1)

        auc = roc_auc_score(y_tes, y_pred)
        aupr = average_precision_score(y_tes, y_pred)
        f1 = f1_score(y_tes, np.round(y_pred.reshape(-1)))
        result = f"auc:{format(auc, '.4f')}\naupr:{format(aupr, '.4f')}\nf1:{format(f1, '.4f')}"
        print("AUC : ", auc)
        print("AUPR : ", aupr)
        print("f1 : ", f1)
        row.append(result)
    rows.append(row)


def writeCSV(header, rows, file_name):
    with open(file_name, 'wt', newline='')as f:
        f_csv = csv.writer(f, delimiter=",")
        f_csv.writerow(header)
        f_csv.writerows(rows)
        f.close()
    print(file_name, "write over!!!")


writeCSV(header, rows, "specific_result_%s_1.csv" % (index))
