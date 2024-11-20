import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# Kolmogorov-Arnold Network (KAN) 定义
class KolmogorovArnoldNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(KolmogorovArnoldNetwork, self).__init__()
        self.lambdas = nn.Parameter(torch.randn(hidden_dim, input_dim))  # 权重矩阵
        self.phi = nn.ReLU()  # 激活函数，选择 ReLU 作为示例
        self.output_weights = nn.Linear(hidden_dim, output_dim)  # 输出层

    def forward(self, x):
        # 计算单变量函数组合
        hidden_output = self.phi(torch.matmul(x, self.lambdas.T))  # 输入与权重矩阵的乘积后激活
        output = self.output_weights(hidden_output)  # 输出线性组合
        return output

# 修改后的 GINConvNet 模型（二分类）
class GINConvNetWithKAN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GINConvNetWithKAN, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        # GINConv layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # xt处理部分
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=5, padding=2)
        self.fc1_xt = nn.Linear(32 * 1000, output_dim)

        # 使用 KAN 网络替换原来的 MLP 部分
        self.kan_layer = KolmogorovArnoldNetwork(input_dim=256, hidden_dim=256, output_dim=256)

        self.out = nn.Linear(256, 1)  # 输出层

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        # GINConv Layers
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        # 处理 target (xt) 数据
        embedded_xt = self.embedding_xt(target)
        embedded_xt = embedded_xt.permute(0, 2, 1)  # 调整维度以匹配 Conv1d 的输入格式
        conv_xt = self.conv_xt_1(embedded_xt)
        xt = conv_xt.view(-1, 32 * 1000)
        xt = self.fc1_xt(xt)

        # 合并 xd 和 xt
        xc = torch.cat((x, xt), 1)

        # 使用 KAN 网络进行处理
        xc = self.kan_layer(xc)

        # 最终输出层
        out = self.out(xc)
        return out
