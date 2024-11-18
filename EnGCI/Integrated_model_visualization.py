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
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

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

# 定义标准化方法
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

# 创建ESM模型的测试数据集和数据加载器
csv_path_test = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/GPCR_test.csv"
test_dataset_esm = GPCRDatasetESM(csv_path_test, loaded_drug_smile_feature_dict, loaded_sequence_feature_dict)
test_loader_esm = DataLoader(test_dataset_esm, batch_size=128, shuffle=False)

# 提取ESM模型的中间特征
def predicting_esm_with_features(model_esm, device, loader_esm):
    model_esm.eval()
    total_preds = torch.Tensor().to(device)
    total_labels = torch.Tensor().to(device)
    total_features = []
    
    with torch.no_grad():
        for batch_esm in loader_esm:
            drug_features, protein_features, labels = batch_esm
            drug_features = drug_features.to(device)
            protein_features = protein_features.to(device)
            labels = labels.to(device)

            # 获取ESM模型的输出和中间层特征
            outputs_esm, esm_features = model_esm(drug_features, protein_features, return_features=True)
            outputs_esm_prob = torch.sigmoid(outputs_esm).squeeze()
            outputs_esm_prob = outputs_esm_prob.to(device)

            if outputs_esm_prob.dim() == 0:
                outputs_esm_prob = outputs_esm_prob.unsqueeze(0)

            total_preds = torch.cat((total_preds, outputs_esm_prob), 0)
            total_labels = torch.cat((total_labels, labels), 0)
            total_features.append(esm_features.cpu().detach().numpy())  # 保存中间层特征
    
    total_features = np.vstack(total_features)  # 将特征合并成一个numpy数组
    return total_labels.cpu().numpy().flatten(), total_preds.cpu().numpy().flatten(), total_features

# 提取GIN模型的中间特征
def predicting_gin_with_features(model_gin, device, loader_gin):
    model_gin.eval()
    total_preds = torch.Tensor().to(device)
    total_labels = torch.Tensor().to(device)
    total_features = []
    
    with torch.no_grad():
        for data_gin in loader_gin:
            data_gin = data_gin.to(device)
            labels = data_gin.y.to(device)

            # 获取GIN模型的输出和中间层特征
            outputs_gin, gin_features = model_gin(data_gin)  # 不再传递 return_features 参数
            outputs_gin_prob = torch.sigmoid(outputs_gin).squeeze()
            outputs_gin_prob = outputs_gin_prob.to(device)

            if outputs_gin_prob.dim() == 0:
                outputs_gin_prob = outputs_gin_prob.unsqueeze(0)

            total_preds = torch.cat((total_preds, outputs_gin_prob), 0)
            total_labels = torch.cat((total_labels, labels), 0)
            total_features.append(gin_features.cpu().detach().numpy())  # 保存中间层特征

    total_features = np.vstack(total_features)  # 将特征合并成一个numpy数组
    return total_labels.cpu().numpy().flatten(), total_preds.cpu().numpy().flatten(), total_features

# # t-SNE 可视化函数，删除 epoch 参数，适用于验证阶段
# def plot_tsne_combined(features, labels, save_dir="/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/t_sne_merge_3"):
#     X = features
#     y = labels.astype(int)
#     tsne = TSNE(n_components=2, init='random', random_state=1, learning_rate=200.0)
#     X_tsne = tsne.fit_transform(X)

#     x_min, x_max = X_tsne.min(0), X_tsne.max(0)
#     X_norm = X_tsne

#     plt.figure(figsize=(15, 15))
#     plt.xlim([x_min[0] - 5, x_max[0] + 5])
#     plt.ylim([x_min[1] - 5, x_max[1] + 5])

#     # 定义颜色
#     colors = ['#ffdd89', '#87ceeb']
    
#     # 绘制散点图
#     for i in range(len(X_norm)):
#         plt.scatter(X_norm[i, 0], X_norm[i, 1], color=colors[y[i]], alpha=0.7)

#     # 添加图例，并调整图例字体大小
#     scatter1 = plt.scatter([], [], color=colors[0], label='negative pair', alpha=0.7)
#     scatter2 = plt.scatter([], [], color=colors[1], label='positive pair', alpha=0.7)
#     plt.legend(handles=[scatter1, scatter2], loc='upper left', fontsize=16)

#     # 设置标题并调整字体大小
#     plt.title(f'Combined Model scatter plot', fontsize=20)

#     # 调整坐标轴刻度字体大小
#     plt.tick_params(axis='both', which='major', labelsize=16)
    
#     # 保存图像为 .png 文件，并去除白边
#     plt.savefig(os.path.join(save_dir, f"combined_model_scatter_plot.png"), bbox_inches='tight')
#     plt.close()

# 使用 PCA 进行降维
def plot_pca_combined(features, labels, save_dir="/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/t_sne_merge_3"):
    X = features
    y = labels.astype(int)
    
    pca = PCA(n_components=2)  # 将高维数据降到二维
    X_pca = pca.fit_transform(X)

    x_min, x_max = X_pca.min(0), X_pca.max(0)
    X_norm = X_pca

    plt.figure(figsize=(15, 15))
    plt.xlim([x_min[0] - 5, x_max[0] + 5])
    plt.ylim([x_min[1] - 5, x_max[1] + 5])

    colors = ['#ffdd89', '#87ceeb']
    for i in range(len(X_norm)):
        plt.scatter(X_norm[i, 0], X_norm[i, 1], color=colors[y[i]], alpha=0.7)

    scatter1 = plt.scatter([], [], color=colors[0], label='negative pair', alpha=0.7)
    scatter2 = plt.scatter([], [], color=colors[1], label='positive pair', alpha=0.7)
    plt.legend(handles=[scatter1, scatter2], loc='upper left', fontsize=16)

    plt.title(f'Intermediate Feature Model Scatter Plot', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    plt.savefig(os.path.join(save_dir, f"combined_model_pca_scatter_plot.png"), bbox_inches='tight')
    plt.close()

# K-Means 聚类和 PCA 可视化
# def plot_pca_with_kmeans(features, labels, n_clusters=2, save_dir="/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/t_sne_merge_3"):
#     X = features
#     y = labels.astype(int)
    
#     # 使用KMeans进行聚类
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     cluster_labels = kmeans.fit_predict(X)
    
#     # 计算聚类的轮廓系数
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     print(f'Silhouette Score: {silhouette_avg:.4f}')

#     # PCA降维
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X)

#     x_min, x_max = X_pca.min(0), X_pca.max(0)
#     X_norm = X_pca

#     plt.figure(figsize=(15, 15))
#     plt.xlim([x_min[0] - 5, x_max[0] + 5])
#     plt.ylim([x_min[1] - 5, x_max[1] + 5])

#     # 使用聚类标签进行颜色区分
#     colors = ['#ffdd89', '#87ceeb', '#ff6347', '#32cd32']  # 根据聚类数n_clusters调整颜色数量
#     for i in range(len(X_norm)):
#         plt.scatter(X_norm[i, 0], X_norm[i, 1], color=colors[cluster_labels[i] % len(colors)], alpha=0.7)

#     scatter1 = plt.scatter([], [], color=colors[0], label='negative pair', alpha=0.7)
#     scatter2 = plt.scatter([], [], color=colors[1], label='positive pair', alpha=0.7)
#     plt.legend(handles=[scatter1, scatter2], loc='upper left', fontsize=16)

#     plt.title(f'Ensemble model scatter plot', fontsize=20)
#     plt.tick_params(axis='both', which='major', labelsize=16)
    
#     plt.savefig(os.path.join(save_dir, f"pca_kmeans_clusters_{n_clusters}.png"), bbox_inches='tight')
#     plt.close()

# # 使用 UMAP 进行降维并绘制图像
# def plot_umap_combined(features, labels, save_dir="/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/t_sne_merge_3"):
#     # 检查并创建保存目录
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     X = features
#     y = labels.astype(int)
    
#     # 使用UMAP进行降维，仅调用一次fit_transform
#     umap_model = umap.UMAP(n_components=2, random_state=42)
#     X_umap = umap_model.fit_transform(X)

#     x_min, x_max = X_umap.min(0), X_umap.max(0)
#     X_norm = X_umap

#     plt.figure(figsize=(15, 15))
#     plt.xlim([x_min[0] - 5, x_max[0] + 5])
#     plt.ylim([x_min[1] - 5, x_max[1] + 5])

#     colors = ['#ffdd89', '#87ceeb']
#     for i in range(len(X_norm)):
#         plt.scatter(X_norm[i, 0], X_norm[i, 1], color=colors[y[i]], alpha=0.7)

#     scatter1 = plt.scatter([], [], color=colors[0], label='negative pair', alpha=0.7)
#     scatter2 = plt.scatter([], [], color=colors[1], label='positive pair', alpha=0.7)
#     plt.legend(handles=[scatter1, scatter2], loc='upper left', fontsize=16)

#     plt.title(f'Combined Model UMAP scatter plot', fontsize=20)
#     plt.tick_params(axis='both', which='major', labelsize=16)

#     # 保存图像为 .png 文件，并去除白边
#     plt.savefig(os.path.join(save_dir, f"combined_model_umap_scatter_plot.png"), bbox_inches='tight')
#     plt.close()

# 加载模型和测试数据集，并进行预测
G_test, P_esm_test, esm_features = predicting_esm_with_features(model_esm, device, test_loader_esm)
_, P_gin_test, gin_features = predicting_gin_with_features(model_gin, device, test_loader_gin)

# 标准化GIN和ESM特征
scaler = StandardScaler()
esm_features_scaled = scaler.fit_transform(esm_features)
gin_features_scaled = scaler.fit_transform(gin_features)

# 再进行拼接
combined_features_scaled = np.concatenate([esm_features_scaled, gin_features_scaled], axis=1)

# # 使用KMeans进行聚类并可视化
# n_clusters = 2  # 你可以根据数据的复杂性调整聚类数
# plot_pca_with_kmeans(combined_features_scaled, G_test, n_clusters=n_clusters)

plot_pca_combined(combined_features_scaled, G_test)

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

# 使用最佳权重组合预测结果
best_weights = (0.1, 0.9)  # 从之前的搜索中得到最佳权重
P_combined_test = best_weights[0] * P_esm_test + best_weights[1] * P_gin_test

# 计算最终的AUC和其他评估指标
auc_score, prc_score, precision, recall, accuracy = calculate_metrics(G_test, P_combined_test)
print(f'Final Combined Model - AUC: {auc_score:.4f}, PRC: {prc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')

# 保存最终的评估结果
result_file_name_combined = "eval_combined_results.csv"
with open(result_file_name_combined, 'w') as f:
    f.write(','.join(map(str, [auc_score, prc_score, precision, recall, accuracy])))
print(f'Results saved to {result_file_name_combined}')