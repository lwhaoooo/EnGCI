import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from models.esm_uni_mol import AffinityModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

# 指定保存文件的目录和文件名____训练集
train_save_dir_esm = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/processed/esm_dict"
train_save_filename_esm = "sequence_feature_dict_train.pkl"
train_save_path_esm = os.path.join(train_save_dir_esm, train_save_filename_esm)

train_save_dir_uni_mol = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/processed/uni-mol_dict"
train_save_filename_uni_mol= "drug_smile_feature_dict_train.pkl"
train_save_path_uni_mol = os.path.join(train_save_dir_uni_mol, train_save_filename_uni_mol)

# 确保保存目录存在
if not os.path.exists(train_save_path_esm):
    raise FileNotFoundError(f"The file {train_save_path_esm} does not exist.")

if not os.path.exists(train_save_path_uni_mol):
    raise FileNotFoundError(f"The file {train_save_path_uni_mol} does not exist.")

# 加载字典文件
with open(train_save_path_esm, "rb") as f:
    train_loaded_sequence_feature_dict = pickle.load(f)

with open(train_save_path_uni_mol, "rb") as f:
    train_loaded_drug_smile_feature_dict = pickle.load(f)

# 指定保存文件的目录和文件名____测试集
test_save_dir_esm = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/processed/esm_dict"
test_save_filename_esm = "sequence_feature_dict_test.pkl"
test_save_path_esm = os.path.join(test_save_dir_esm, test_save_filename_esm)

test_save_dir_uni_mol = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/processed/uni-mol_dict"
test_save_filename_uni_mol= "drug_smile_feature_dict_test.pkl"
test_save_path_uni_mol = os.path.join(test_save_dir_uni_mol, test_save_filename_uni_mol)

# 确保保存目录存在
if not os.path.exists(test_save_path_esm):
    raise FileNotFoundError(f"The file {test_save_path_esm} does not exist.")

if not os.path.exists(test_save_path_uni_mol):
    raise FileNotFoundError(f"The file {test_save_path_uni_mol} does not exist.")

# 加载字典文件
with open(test_save_path_esm, "rb") as f:
    test_loaded_sequence_feature_dict = pickle.load(f)

with open(test_save_path_uni_mol, "rb") as f:
    test_loaded_drug_smile_feature_dict = pickle.load(f)

def standardize(tensor):
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True)
    standardized_tensor = (tensor - mean) / std
    return standardized_tensor

# 对训练集中字典中的张量进行归一化
for key, value in train_loaded_drug_smile_feature_dict.items():
    train_loaded_drug_smile_feature_dict[key] = standardize(torch.tensor(value).clone().detach())

for key, value in train_loaded_sequence_feature_dict.items():
    train_loaded_sequence_feature_dict[key] = standardize(value.clone().detach())

# 对测试集中字典中的张量进行归一化
for key, value in test_loaded_drug_smile_feature_dict.items():
    test_loaded_drug_smile_feature_dict[key] = standardize(torch.tensor(value).clone().detach())

for key, value in test_loaded_sequence_feature_dict.items():
    test_loaded_sequence_feature_dict[key] = standardize(value.clone().detach())

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GPCRDataset(Dataset):
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
        
        label = torch.tensor(label, dtype=torch.float32)
        
        return drug_feature, protein_feature, label

# 创建数据集
csv_path_train = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/GPCR_train.csv"
csv_path_test = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/GPCR_test.csv"
train_dataset = GPCRDataset(csv_path_train, train_loaded_drug_smile_feature_dict, train_loaded_sequence_feature_dict)
test_dataset = GPCRDataset(csv_path_test, test_loaded_drug_smile_feature_dict, test_loaded_sequence_feature_dict)

# 将训练集划分为80%训练集和20%验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# 创建DataLoader
train_batch_size = 512
test_batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)  # 验证集
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)

model = AffinityModel().to(device)
criterion = nn.BCEWithLogitsLoss()  # 修改：使用 BCEWithLogitsLoss 作为损失函数

# 修改1：增加学习率调度器
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)  # 添加 L2 正则化
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)  # 学习率调度器

def calculate_metrics(y_true, y_pred):
    y_pred_labels = (y_pred >= 0.5).astype(int)  # 修改：将概率转换为二值标签
    auc = roc_auc_score(y_true, y_pred)
    prc = average_precision_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    accuracy = accuracy_score(y_true, y_pred_labels)
    
    return auc, prc, precision, recall, accuracy

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor().to(device)
    total_labels = torch.Tensor().to(device)
    with torch.no_grad():
        for drug_features, protein_features, labels in loader:
            drug_features = drug_features.to(device)
            protein_features = protein_features.to(device)
            labels = labels.to(device)
            outputs = model(drug_features, protein_features)
            preds = torch.sigmoid(outputs).squeeze(1)  # 修改：使用 Sigmoid 将输出转为概率值，并调整维度
            total_preds = torch.cat((total_preds, preds), dim=0)
            total_labels = torch.cat((total_labels, labels), dim=0)
    return total_labels.cpu().numpy().flatten(), total_preds.cpu().numpy().flatten()

# def predicting(model, device, loader):
#     model.eval()
#     total_preds = torch.Tensor().to(device)
#     total_labels = torch.Tensor().to(device)
#     with torch.no_grad():
#         for drug_features, protein_features, labels in loader:
#             drug_features = drug_features.to(device)
#             protein_features = protein_features.to(device)
#             labels = labels.to(device)
            
#             # 调用模型并获取输出
#             outputs = model(drug_features, protein_features)
#             preds = torch.sigmoid(outputs).squeeze(1)  # 使用 Sigmoid 将输出转为概率值，并调整维度
            
#             total_preds = torch.cat((total_preds, preds), dim=0)
#             total_labels = torch.cat((total_labels, labels), dim=0)
    
#     return total_labels.cpu().numpy().flatten(), total_preds.cpu().numpy().flatten()


LOG_INTERVAL = 20  # 定义每20次打印一次

# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (drug_features, protein_features, labels) in enumerate(train_loader):
        drug_features = drug_features.to(device)
        protein_features = protein_features.to(device)
        labels = labels.to(device).float()  # 修改：将标签转换为 float 类型
        
        optimizer.zero_grad()
        # outputs, _ = model(drug_features, protein_features)  # 忽略中间特征
        outputs = model(drug_features, protein_features).squeeze(1)  # 修改：调整输出维度以匹配 BCEWithLogitsLoss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 每20个批次打印一次损失
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(drug_features),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    
    # 在每个epoch结束时打印平均损失
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1} average loss: {avg_loss:.6f}')

# def train(model, device, train_loader, optimizer, epoch):
#     model.train()
#     running_loss = 0.0
#     for batch_idx, (drug_features, protein_features, labels) in enumerate(train_loader):
#         drug_features = drug_features.to(device)
#         protein_features = protein_features.to(device)
#         labels = labels.to(device).float()  # 修改：将标签转换为 float 类型
        
#         optimizer.zero_grad()

#         # 确保模型输出赋值到 outputs
#         outputs = model(drug_features, protein_features)  # 使用模型预测
#         outputs = outputs.squeeze(1)  # 修改：调整输出维度以匹配 BCEWithLogitsLoss
        
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
        
#         # 每20个批次打印一次损失
#         if batch_idx % LOG_INTERVAL == 0:
#             print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
#                                                                            batch_idx * len(drug_features),
#                                                                            len(train_loader.dataset),
#                                                                            100. * batch_idx / len(train_loader),
#                                                                            loss.item()))
    
#     # 在每个epoch结束时打印平均损失
#     avg_loss = running_loss / len(train_loader)
#     print(f'Epoch {epoch + 1} average loss: {avg_loss:.6f}')


# 训练循环
num_epochs = 500
best_auc = 0.0
best_model_path = "model_esm_uni-mol_best_eval.model"
result_file_name = "result_esm_uni-mol_best_eval.csv"
best_epoch = 0

print(f'Training on {len(train_loader.dataset)} samples...')

for epoch in range(num_epochs):
    train(model, device, train_loader, optimizer, epoch)
    
    # 在验证集上进行预测
    print(f"Evaluating on validation set at epoch {epoch + 1}...")
    y_true_val, y_pred_val = predicting(model, device, val_loader)
    auc_val, prc_val, precision_val, recall_val, accuracy_val = calculate_metrics(y_true_val, y_pred_val)
    print(f'Epoch {epoch+1}: Validation AUC: {auc_val:.4f}, PRC: {prc_val:.4f}, Precision: {precision_val:.4f}, Recall: {recall_val:.4f}, Accuracy: {accuracy_val:.4f}')
    
    # 修改3：调整学习率调度器
    scheduler.step(auc_val)  # 使用验证集的 AUC 来调整学习率

    # 保存最佳模型
    if auc_val > best_auc:
        best_auc = auc_val
        best_epoch = epoch + 1  # 更新最佳AUC对应的epoch
        torch.save(model.state_dict(), best_model_path)
        with open(result_file_name, 'w') as f:
            f.write(','.join(map(str, [auc_val, prc_val, precision_val, recall_val, accuracy_val])))
        print(f"AUC improved at epoch {epoch+1}; best_auc: {best_auc:.4f}")
    else:
        print(f"No improvement since epoch {best_epoch}; best_auc: {best_auc:.4f}")

print(f"Best metrics saved to {result_file_name}")