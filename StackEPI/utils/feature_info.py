import os
import sys

import numpy as np

# print("python搜索模块的路径集合：", )
# for path in sys.path:
#     print(path)
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])

from utils.utils_data import get_data_np_dict

eiip_data = get_data_np_dict("epivan", "HUVEC", "pseknc", "a")

print(eiip_data["train_X"].shape)
print(eiip_data["train_y"].shape)
print(eiip_data["test_X"].shape)
print(eiip_data["test_y"].shape)

# kmer_data = get_data_np_dict("epivan", "GM12878", "kmer", "a")
# eiip_data = get_data_np_dict("epivan", "GM12878", "eiip", "a")
# print(kmer_data["train_X"].shape)
# print(eiip_data["train_X"].shape)
# print(np.array(kmer_data["train_X"]).mean(axis=0))
# print(np.array(kmer_data["train_X"]).std(axis=0))
# print(np.array(eiip_data["train_X"][0]).mean(axis=0))
# print(np.array(eiip_data["train_X"][0]).std(axis=0))
# print(kmer_data["train_X"][0] - eiip_data["train_X"][0])
