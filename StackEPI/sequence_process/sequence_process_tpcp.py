import os
import sys

root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])
from sequence_process.DPCP import DPCP
import numpy as np

from sequence_process.sequence_process_def import get_cell_line_seq


names = ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']
cell_name = names[4]
feature_name = "tpcp"
data_source = "epivan"

feature_dir, \
enhancers_tra, promoters_tra, y_tra, \
im_enhancers_tra, im_promoters_tra, \
y_imtra, enhancers_tes, promoters_tes, y_tes = get_cell_line_seq(data_source, cell_name, feature_name)




set_pc_list = ["Bendability-DNAse",
               "Bendability-consensus",
               "Trinucleotide GC Content",
               "Nucleosome positioning",
               "Consensus_roll",
               "Dnase I",
               "Dnase I-Rigid",
               "MW-Daltons",
               "Nucleosome",
               "Nucleosome-Rigid",
               ]


def get_data(enhancers, promoters):
    dpcp = DPCP(3, set_pc_list, n_jobs=1)
    X_en = dpcp.run_DPCP(enhancers)
    # print(X_en)
    X_pr = dpcp.run_DPCP(promoters)
    # print(X_pr)
    return np.array(X_en), np.array(X_pr)


"""
get and save
"""
X_en_tra, X_pr_tra = get_data(enhancers_tra, promoters_tra)
np.savez(feature_dir + '%s_train.npz' % cell_name, X_en_tra=X_en_tra, X_pr_tra=X_pr_tra, y_tra=y_tra)
if data_source == "epivan":
    X_en_imtra, X_pr_imtra = get_data(im_enhancers_tra, im_promoters_tra)
    np.savez(feature_dir + 'im_%s_train.npz' % cell_name, X_en_tra=X_en_imtra, X_pr_tra=X_pr_imtra, y_tra=y_imtra)
X_en_tes, X_pr_tes = get_data(enhancers_tes, promoters_tes)
np.savez(feature_dir + '%s_test.npz' % cell_name, X_en_tes=X_en_tes, X_pr_tes=X_pr_tes, y_tes=y_tes)
print("save over!")


