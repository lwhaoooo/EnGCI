import numpy as np
import pandas as pd
import sys, os  # 用于系统和文件操作
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv_GPCR_new_3conv import GINConvNet
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
import pdb

# 在每个 epoch 训练模型的函数
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()  # 将模型设置为训练模式
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)  # 将数据移到指定的设备上
        optimizer.zero_grad()  # 清除梯度
        output = model(data).view(-1)  # 修改：调整输出维度以匹配 BCEWithLogitsLoss
        loss = loss_fn(output, data.y.float().to(device))  # 修改：使用 BCEWithLogitsLoss 需要将标签转为 float 类型
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

# def train(model, device, train_loader, optimizer, epoch):
#     print('Training on {} samples...'.format(len(train_loader.dataset)))
#     model.train()  # 将模型设置为训练模式
#     for batch_idx, data in enumerate(train_loader):
#         data = data.to(device)  # 将数据移到指定的设备上
#         optimizer.zero_grad()  # 清除梯度
#         output, _ = model(data)  # 忽略中间特征，只需要输出
#         output = output.view(-1)  # 修改：调整输出维度以匹配 BCEWithLogitsLoss
#         loss = loss_fn(output, data.y.float().to(device))  # 修改：使用 BCEWithLogitsLoss 需要将标签转为 float 类型
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新模型参数
#         if batch_idx % LOG_INTERVAL == 0:
#             print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
#                                                                            batch_idx * len(data.x),
#                                                                            len(train_loader.dataset),
#                                                                            100. * batch_idx / len(train_loader),
#                                                                            loss.item()))

# def predicting(model, device, loader):
#     model.eval()  # 将模型设置为评估模式
#     total_preds = torch.Tensor()  # 存储所有预测结果
#     total_labels = torch.Tensor()  # 存储所有真实标签
#     print('Make prediction for {} samples...'.format(len(loader.dataset)))
#     with torch.no_grad():  # 关闭梯度计算
#         for data in loader:
#             data = data.to(device)  # 将数据移到指定的设备上
#             output, _ = model(data)  # 忽略中间特征
#             output = output.view(-1)  # 修改：调整输出维度以匹配 BCEWithLogitsLoss
#             preds = torch.sigmoid(output)  # 修改：使用 Sigmoid 将输出转为概率值
#             preds = (preds > 0.5).float()  # 修改：将概率值转为二分类结果
#             total_preds = torch.cat((total_preds, preds.cpu()), 0)  # 记录预测结果
#             total_labels = torch.cat((total_labels, data.y.cpu()), 0)  # 记录真实标签
#     return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def predicting(model, device, loader):
    model.eval()  # 将模型设置为评估模式
    total_preds = torch.Tensor()  # 存储所有预测结果
    total_labels = torch.Tensor()  # 存储所有真实标签
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():  # 关闭梯度计算
        for data in loader:
            data = data.to(device)  # 将数据移到指定的设备上
            output = model(data).view(-1)  # 修改：调整输出维度以匹配 BCEWithLogitsLoss
            preds = torch.sigmoid(output)  # 修改：使用 Sigmoid 将输出转为概率值
            preds = (preds > 0.5).float()  # 修改：将概率值转为二分类结果
            total_preds = torch.cat((total_preds, preds.cpu()), 0)  # 记录预测结果
            total_labels = torch.cat((total_labels, data.y.cpu()), 0)  # 记录真实标签
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def evaluate_performance(labels, preds):
    auc = roc_auc_score(labels, preds)
    prc = average_precision_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    accuracy = accuracy_score(labels, preds)
    return auc, prc, precision, recall, accuracy

datasets = [['GPCR']][int(sys.argv[1])]  # 选择数据集
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]  # 选择模型
model_st = modeling.__name__

cuda_name = "cuda:0"  # 默认使用 cuda:0 作为 GPU 设备
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512  # 训练批次大小
TEST_BATCH_SIZE = 512  # 测试批次大小
LR = 0.001  # 学习率
LOG_INTERVAL = 20  # 日志记录间隔
NUM_EPOCHS = 1000  # 训练的 epoch 数

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# 主程序：迭代处理不同的数据集
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset)
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset+'_train')  # 加载预处理后的数据
        test_data = TestbedDataset(root='data', dataset=dataset+'_test')

        # 将数据划分为训练集和验证集 (80% 训练集，20% 验证集)
        train_size = int(0.8 * len(train_data))  # 80% 作为训练集
        valid_size = len(train_data) - train_size  # 剩余 20% 作为验证集
        train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

        # 将数据准备为 PyTorch mini-batch 处理格式
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # 训练模型
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)  # 初始化模型并移到指定设备上
        loss_fn = nn.BCEWithLogitsLoss()  # 使用 BCEWithLogitsLoss 作为损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 使用 Adam 优化器
        best_auc = 0  # 初始化最佳AUC
        best_epoch = -1  # 初始化最佳epoch
        model_file_name = 'model_' + model_st + '_' + dataset + '_eval' + '.model'  # 模型文件名
        result_file_name = 'result_' + model_st + '_' + dataset +'_eval' + '.csv'  # 结果文件名

        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch+1)  # 训练模型
            print('Validating for validation data')
            G, P = predicting(model, device, valid_loader)  # 在验证集上进行预测
            auc_score, prc_score, precision, recall, accuracy = evaluate_performance(G, P)  # 计算评估指标
            print(f'Epoch {epoch+1}: AUC: {auc_score:.4f}, PRC: {prc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')
            
            # 保存最佳模型
            if auc_score > best_auc:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name, 'w') as f:
                    f.write(','.join(map(str, [auc_score, prc_score, precision, recall, accuracy])))
                best_auc = auc_score
                best_epoch = epoch + 1
                print('AUC improved at epoch ', best_epoch, '; best_auc:', best_auc, model_st, dataset)
            else:
                print('No improvement since epoch ', best_epoch, '; best_auc:', best_auc, model_st, dataset)