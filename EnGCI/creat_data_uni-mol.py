import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pickle
from unimol_tools import UniMolRepr
import pdb
import os

# 定义文件路径
file_path_train = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/GPCR_train.csv"
file_path_test = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/GPCR_test.csv"

# 读取数据
data = pd.read_csv(file_path_test)

# 药物分子序列去重
data_cleaned_smile = data.drop_duplicates(subset=['compound_iso_smiles'])

# 转化为列表
compound_smile = list(data_cleaned_smile['compound_iso_smiles'])

# 定义 UniMolRepr 对象
clf = UniMolRepr(data_type='molecule', remove_hs=False)

# 定义一个 PyTorch Dataset 类
class MoleculeDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        return self.smiles_list[idx]

# 创建 Dataset 和 DataLoader
dataset = MoleculeDataset(compound_smile)
batch_size = 64  # 你可以根据你的显存情况调整批量大小
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 使用 DataLoader 批量处理数据并显示进度条
drug_smile_feature_dict = {}
batch_index = 0

for batch in tqdm(dataloader, desc="Processing molecules"):
    start_idx = batch_index * batch_size
    end_idx = start_idx + len(batch)  # 确保最后一个批次不会超出范围
    batch_smiles = compound_smile[start_idx:end_idx]
    # pdb.set_trace()
    batch_reprs = clf.get_repr(batch, return_atomic_reprs=True)['cls_repr']
    batch_reprs = np.array(batch_reprs)
    
    for smile, feature in zip(batch_smiles, batch_reprs):
        drug_smile_feature_dict[smile] = feature
    
    batch_index += 1

# 打印字典中的部分内容以进行检查
for seq, feature in list(drug_smile_feature_dict.items())[:5]:
    print(f"smile: {seq}, Feature vector shape: {feature.shape}")

# 指定保存文件的目录和文件名
save_dir = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/processed/uni-mol_dict"
save_filename = "drug_smile_feature_dict_test.pkl"

save_path = os.path.join(save_dir, save_filename)

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# 将结果保存到指定位置
with open(save_path, 'wb') as f:
    pickle.dump(drug_smile_feature_dict, f)

print(f"Dictionary saved to {save_path}")