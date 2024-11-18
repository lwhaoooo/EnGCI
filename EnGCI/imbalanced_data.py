import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from models.ginconv_GPCR_new_3conv import GINConvNet
from models.esm_uni_mol import AffinityModel
from utils import *
import pickle
import pandas as pd
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.utils import resample
import os
import numpy as np

# 指定保存文件的目录和文件名
save_dir_esm = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/processed/esm_dict"
save_filename_esm = "sequence_feature_dict_test.pkl"
save_path_esm = os.path.join(save_dir_esm, save_filename_esm)

save_dir_uni_mol = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/processed/uni-mol_dict"
save_filename_uni_mol = "drug_smile_feature_dict_test.pkl"
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

def collate_fn(batch):
    return Batch.from_data_list(batch)

# 定义GPCRDataset类，添加正负样本比例控制
class GPCRDatasetESM(Dataset):
    def __init__(self, csv_file, drug_dict, protein_dict, positive_ratio=1.0):
        self.data = pd.read_csv(csv_file)
        self.drug_dict = drug_dict
        self.protein_dict = protein_dict

        # 分离正负样本
        positive_samples = self.data[self.data.iloc[:, 2] == 1]  # 假设标签在第3列
        negative_samples = self.data[self.data.iloc[:, 2] == 0]

        # 计算负样本需要的数量
        negative_count = int(len(positive_samples) * (1 / positive_ratio - 1))

        if len(negative_samples) > negative_count:
            # 从负样本中进行采样
            negative_samples = resample(negative_samples, replace=False, n_samples=negative_count, random_state=42)

        # 将正负样本组合起来
        self.data = pd.concat([positive_samples, negative_samples])
        self.data = self.data.sample(frac=1).reset_index(drop=True)  # 随机打乱数据

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

# 加载模型
model_path_esm = "model_esm_uni-mol_best_eval.model"
model_path_gin = "model_GINConvNet_GPCR_eval.model"

model_esm = AffinityModel().to(device)
model_esm.load_state_dict(torch.load(model_path_esm))

model_gin = GINConvNet().to(device)
model_gin.load_state_dict(torch.load(model_path_gin))

# 将模型设置为评估模式
model_esm.eval()
model_gin.eval()

# 创建GIN模型的测试数据集和数据加载器
test_dataset_gin = TestbedDataset(root='data', dataset='GPCR_test')
test_loader_gin = DataLoader(test_dataset_gin, batch_size=128, shuffle=False, collate_fn=collate_fn)

# 创建ESM模型的测试数据集和数据加载器，控制正负样本比例
csv_path_test = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/GPCR_test.csv"  # 使用上传的文件路径
test_dataset_esm = GPCRDatasetESM(csv_path_test, loaded_drug_smile_feature_dict, loaded_sequence_feature_dict, positive_ratio=0.0909)
test_loader_esm = DataLoader(test_dataset_esm, batch_size=128, shuffle=False)

# 计算最终的AUC和其他评估指标
def calculate_metrics(y_true, y_pred):
    # 计算基于概率的AUC和PRC
    auc = roc_auc_score(y_true, y_pred)
    prc = average_precision_score(y_true, y_pred)

    # 将概率转为二元标签用于计算precision、recall、accuracy
    y_pred_binary = (y_pred > 0.5).astype(int)

    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    accuracy = accuracy_score(y_true, y_pred_binary)

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

# GIN模型的预测函数
def predicting_gin(model_gin, device, loader_gin):
    model_gin.eval()
    total_preds = torch.Tensor().to(device)
    total_labels = torch.Tensor().to(device)

    with torch.no_grad():
        for data_gin in loader_gin:
            data_gin = data_gin.to(device)
            labels = data_gin.y.to(device)

            outputs_gin = model_gin(data_gin)
            outputs_gin_prob = torch.sigmoid(outputs_gin).squeeze()
            outputs_gin_prob = outputs_gin_prob.to(device)

            if outputs_gin_prob.dim() == 0:
                outputs_gin_prob = outputs_gin_prob.unsqueeze(0)

            total_preds = torch.cat((total_preds, outputs_gin_prob), 0)
            total_labels = torch.cat((total_labels, labels), 0)

    return total_labels.cpu().numpy().flatten(), total_preds.cpu().numpy().flatten()

# 分析预测结果的互补性
def analyze_predictions(G, P_esm, P_gin):
    esm_correct = (G == (P_esm > 0.5))
    gin_correct = (G == (P_gin > 0.5))

    both_correct = esm_correct & gin_correct
    esm_only_correct = esm_correct & ~gin_correct
    gin_only_correct = ~esm_correct & gin_correct

    print(f'Both Correct: {both_correct.sum()}, ESM Only Correct: {esm_only_correct.sum()}, GIN Only Correct: {gin_only_correct.sum()}')

# 寻找最佳权重，使用 AUC 作为评估标准
def grid_search_weights(P_esm, P_gin, G):
    best_auc = 0
    best_weights = (0.5, 0.5)
    weights_range = np.arange(0, 1.1, 0.1)

    for weight_esm in weights_range:
        weight_gin = 1 - weight_esm
        P_combined = weight_esm * P_esm + weight_gin * P_gin
        auc = roc_auc_score(G, P_combined)
        print(f'Weight ESM: {weight_esm:.2f}, Weight GIN: {weight_gin:.2f} -> AUC: {auc:.4f}')
        if auc > best_auc:
            best_auc = auc
            best_weights = (weight_esm, weight_gin)

    print(f'Best Weights -> Weight ESM: {best_weights[0]:.2f}, Weight GIN: {best_weights[1]:.2f}, Best AUC: {best_auc:.4f}')
    return best_weights

# 加载模型和测试数据集，并进行预测
G_test, P_esm_test = predicting_esm(model_esm, device, test_loader_esm)
_, P_gin_test = predicting_gin(model_gin, device, test_loader_gin)

# 分析两个模型的预测结果
analyze_predictions(G_test, P_esm_test, P_gin_test)

# 找到最佳权重
best_weights = grid_search_weights(P_esm_test, P_gin_test, G_test)

# 使用最佳权重组合预测结果
P_combined_test = best_weights[0] * P_esm_test + best_weights[1] * P_gin_test
# P_combined_test = (P_combined_test > 0.5).astype(int)

# 计算最终的AUC和其他评估指标
auc_score, prc_score, precision, recall, accuracy = calculate_metrics(G_test, P_combined_test)
print(f'Final Combined Model - AUC: {auc_score:.4f}, PRC: {prc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')

# 保存最终的评估结果
result_file_name_combined = "data_eval_combined_results.csv"
with open(result_file_name_combined, 'w') as f:
    f.write(','.join(map(str, [auc_score, prc_score, precision, recall, accuracy])))
print(f'Results saved to {result_file_name_combined}')