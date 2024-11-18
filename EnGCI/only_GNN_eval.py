import torch
import torch.nn as nn
import sys
from models.ginconv_GPCR_new_3conv import GINConvNet
from models.gat import GATNet  # 根据实际的文件结构调整导入路径
from models.gcn import GCNNet
from models.gat_gcn import GAT_GCN
from utils import *
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
import numpy as np

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    return Batch.from_data_list(batch)

# 选择模型，使用数字进行选择
modeling_classes = [
    GINConvNet,   # 0 对应 GINConvNet
    GCNNet,       # 1 对应 GCNNet
    GATNet,       # 2 对应 GATNet
    GAT_GCN       # 3 对应 GAT_GCN
]

# 获取模型索引
model_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 默认使用0，即GINConvNet
if model_index not in range(len(modeling_classes)):
    print(f"无效的模型索引：{model_index}，请使用 0-3 之间的数字")
    sys.exit(1)

model_class = modeling_classes[model_index]
model_name = model_class.__name__
model_path = f"model_{model_name}_GPCR_eval.model"

# 加载选择的模型
model = model_class().to(device)
model.load_state_dict(torch.load(model_path))

# 将模型设置为评估模式
model.eval()

# 创建模型的测试数据集和数据加载器
test_dataset = TestbedDataset(root='data', dataset='GPCR_test')  # 根据实际路径进行调整
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

# 计算最终的AUC和其他评估指标
def calculate_metrics(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    prc = average_precision_score(y_true, y_pred)
    y_pred_binary = (y_pred > 0.5).astype(int)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    accuracy = accuracy_score(y_true, y_pred_binary)
    return auc, prc, precision, recall, accuracy

# 定义模型的预测函数
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor().to(device)
    total_labels = torch.Tensor().to(device)
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            labels = data.y.to(device)
            outputs = model(data)  # 这里的 outputs 应该是模型的预测结果
            # outputs, _ = model(data)  # 忽略中间特征
            outputs_prob = torch.sigmoid(outputs).squeeze()
            outputs_prob = outputs_prob.to(device)

            if outputs_prob.dim() == 0:
                outputs_prob = outputs_prob.unsqueeze(0)

            total_preds = torch.cat((total_preds, outputs_prob), 0)
            total_labels = torch.cat((total_labels, labels), 0)
    
    return total_labels.cpu().numpy().flatten(), total_preds.cpu().numpy().flatten()

# 进行预测
G_test, P_test = predicting(model, device, test_loader)

# 计算AUC和其他评估指标
auc_score, prc_score, precision, recall, accuracy = calculate_metrics(G_test, P_test)
print(f'{model_name} - AUC: {auc_score:.4f}, PRC: {prc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')

# 保存评估结果
result_file_name = f"eval_{model_name}_results.csv"
with open(result_file_name, 'w') as f:
    f.write(','.join(map(str, [auc_score, prc_score, precision, recall, accuracy])))
print(f'{model_name} Results saved to {result_file_name}')
