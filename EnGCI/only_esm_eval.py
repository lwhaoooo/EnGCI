import torch
import torch.nn as nn
import torch.nn.functional as F
from models.esm_uni_mol import AffinityModel
from utils import *
import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
import numpy as np
import pdb

# 指定保存文件的目录和文件名
save_dir_esm = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/processed/esm_dict"
save_filename_esm = "sequence_feature_dict_test.pkl"
save_path_esm = os.path.join(save_dir_esm, save_filename_esm)

save_dir_uni_mol = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/processed/uni-mol_dict"
save_filename_uni_mol= "drug_smile_feature_dict_test.pkl"
save_path_uni_mol = os.path.join(save_dir_uni_mol, save_filename_uni_mol)

# 确保保存目录存在
if not os.path.exists(save_path_esm):
    raise FileNotFoundError(f"The file {save_path_esm} does not exist.")

if not os.path.exists(save_path_uni_mol):
    raise FileNotFoundError(f"The file {save_path_uni_mol} does not exist.")

# 加载字典文件
with open(save_path_esm, "rb") as f:
    loaded_sequence_feature_dict = pickle.load(f)

with open(save_path_uni_mol, "rb") as f:
    loaded_drug_smile_feature_dict = pickle.load(f)

def standardize(tensor):
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True)
    standardized_tensor = (tensor - mean) / std
    return standardized_tensor

# 对字典中的张量进行归一化
for key, value in loaded_drug_smile_feature_dict.items():
    loaded_drug_smile_feature_dict[key] = standardize(torch.tensor(value).clone().detach())

for key, value in loaded_sequence_feature_dict.items():
    loaded_sequence_feature_dict[key] = standardize(value.clone().detach())

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义GPCRDataset类
class GPCRDatasetESM(Dataset):
    def __init__(self, csv_file, drug_dict, protein_dict):
        self.data = pd.read_csv(csv_file)
        self.drug_dict = drug_dict
        self.protein_dict = protein_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.data.iloc[idx]
        drug_smile = row[0]
        protein_seq = row[1]
        label = row[2]
        
        # 获取药物特征向量
        if drug_smile in self.drug_dict:
            drug_feature = self.drug_dict[drug_smile].clone().detach().float()
        else:
            raise KeyError(f"Drug smile {drug_smile} not found in dictionary.")
        
        # 获取蛋白质特征向量
        if protein_seq in self.protein_dict:
            protein_feature = self.protein_dict[protein_seq].clone().detach().float()
        else:
            raise KeyError(f"Protein sequence {protein_seq} not found in dictionary.")
        
        label = torch.tensor(label, dtype=torch.long)  # 确保标签为长整型
        
        return drug_feature, protein_feature, label

# 加载ESM模型
model_path_esm = "model_esm_uni-mol_best_eval.model"
model_esm = AffinityModel().to(device)
# pdb.set_trace()
model_esm.load_state_dict(torch.load(model_path_esm))


# 将ESM模型设置为评估模式
model_esm.eval()

# 创建ESM模型的测试数据集和数据加载器
csv_path_test = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/GPCR_test.csv"
test_dataset_esm = GPCRDatasetESM(csv_path_test, loaded_drug_smile_feature_dict, loaded_sequence_feature_dict)
test_loader_esm = DataLoader(test_dataset_esm, batch_size=128, shuffle=False)

# 计算最终的AUC和其他评估指标
def calculate_metrics(y_true, y_pred):
    # 计算基于概率的AUC和PRC
    auc = roc_auc_score(y_true, y_pred)  # 基于连续概率计算AUC
    prc = average_precision_score(y_true, y_pred)  # 基于连续概率计算PRC
    
    # 将概率转为二元标签用于计算precision、recall、accuracy
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    precision = precision_score(y_true, y_pred_binary)  # 基于二元标签计算precision
    recall = recall_score(y_true, y_pred_binary)  # 基于二元标签计算recall
    accuracy = accuracy_score(y_true, y_pred_binary)  # 基于二元标签计算accuracy
    
    return auc, prc, precision, recall, accuracy

# 定义ESM模型的预测函数
def predicting_esm(model_esm, device, loader_esm):
    model_esm.eval()
    total_preds = torch.Tensor().to(device)
    total_labels = torch.Tensor().to(device)
    
    with torch.no_grad():
        for batch_esm in loader_esm:
            drug_features, protein_features, labels = batch_esm
            drug_features = drug_features.to(device)
            protein_features = protein_features.to(device)
            labels = labels.to(device)

            outputs_esm = model_esm(drug_features, protein_features)
            outputs_esm_prob = torch.sigmoid(outputs_esm).squeeze()
            outputs_esm_prob = outputs_esm_prob.to(device)

            if outputs_esm_prob.dim() == 0:
                outputs_esm_prob = outputs_esm_prob.unsqueeze(0)

            total_preds = torch.cat((total_preds, outputs_esm_prob), 0)
            total_labels = torch.cat((total_labels, labels), 0)
    
    return total_labels.cpu().numpy().flatten(), total_preds.cpu().numpy().flatten()

# 加载模型并在测试集上进行预测
G_test, P_esm_test = predicting_esm(model_esm, device, test_loader_esm)

# 计算ESM模型的AUC和其他评估指标
auc_score, prc_score, precision, recall, accuracy = calculate_metrics(G_test, P_esm_test)
print(f'ESM Model - AUC: {auc_score:.4f}, PRC: {prc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')

# 保存ESM模型的评估结果
result_file_name_esm = "eval_esm_results.csv"
with open(result_file_name_esm, 'w') as f:
    f.write(','.join(map(str, [auc_score, prc_score, precision, recall, accuracy])))
print(f'ESM Model Results saved to {result_file_name_esm}')
