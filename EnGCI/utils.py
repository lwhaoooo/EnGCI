import os  # 用于文件和目录操作
import numpy as np
from math import sqrt  # 用于计算平方根
from scipy import stats  # 用于统计计算
from torch_geometric.data import InMemoryDataset, DataLoader  # 导入 PyTorch Geometric 的内存数据集和数据加载器
from torch_geometric import data as DATA  # 简化数据操作的引用
import torch
import pdb

# 继承自 InMemoryDataset 的自定义数据集类
class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='GPCR', 
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):

        # root 参数是用于存储预处理数据的根目录，默认为 '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # dataset 参数是数据集名称，默认为 'davis'
        self.dataset = dataset
        # 检查预处理数据文件是否存在
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            # 如果预处理数据文件不存在，进行预处理并保存数据
            self.process(xd, xt, y,smile_graph)
            
            # self.data: 存储图数据的主要数据结构，通常是 torch_geometric.data.Data 对象或它的列表。
            # self.slices: 用于表示数据切片的信息，以便支持分批次加载数据。
            self.data, self.slices = torch.load(self.processed_paths[0])  

    @property
    def raw_file_names(self):
        pass  # 返回原始数据文件的名称列表（未实现）
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']  # 返回预处理数据文件的名称列表

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        # 如果处理目录不存在，则创建，不复写默认为processed
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, y,smile_graph):
        # 确保输入数据长度相同
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            # pdb.set_trace()
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # 使用 rdkit 将 SMILES 转换为分子表示
            c_size, features, edge_index = smile_graph[smiles]
            # 将图数据准备为 PyTorch Geometrics 的 GCN 算法格式
            GCNData = DATA.Data(x=torch.Tensor(features),  # x表示结点特征
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),  # : 将边索引列表 edge_index 转换为长整型张量 (LongTensor)，并进行转置，使其符合 PyTorch Geometric 的要求（形状为 [2, num_edges]）。
                                y=torch.FloatTensor([labels]))  # y是标签信息，这里表示亲和力
            GCNData.target = torch.LongTensor([target])  # 添加目标序列（蛋白质序列）的整数编码
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))  # 设置节点数量属性 c_size 
            # 将图数据、标签和目标序列添加到数据列表中
            data_list.append(GCNData)

        # 如果定义了预过滤器，则应用 
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        # 如果定义了预变换，则应用
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        # pdb.set_trace()    
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # 保存预处理数据
        torch.save((data, slices), self.processed_paths[0])

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))  # 计算均方根误差
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)  # 计算均方误差
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]  # 计算皮尔逊相关系数
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]  # 计算斯皮尔曼相关系数
    return rs
# 计算一致性指数
def ci(y, f):
    ind = np.argsort(y)  # 对 y 进行排序，返回排序后的索引
    y = y[ind]  # 根据排序后的索引对 y 进行排序
    f = f[ind]  # 根据排序后的索引对 f 进行排序
    i = len(y) - 1  # 初始化索引 i 为 y 的最后一个元素的索引
    j = i - 1  # 初始化索引 j 为 i 之前的一个元素的索引
    z = 0.0  # 初始化计数器 z
    S = 0.0  # 初始化一致性计数器 S
    while i > 0:  # 遍历所有索引对 (i, j)
        while j >= 0:
            if y[i] > y[j]:  # 如果 y[i] > y[j]
                z = z + 1  # 更新计数器 z
                u = f[i] - f[j]  # 计算预测值差异
                if u > 0:
                    S = S + 1  # 更新一致性计数器 S
                elif u == 0:
                    S = S + 0.5  # 更新一致性计数器 S
            j = j - 1  # 更新索引 j
        i = i - 1  # 更新索引 i
        j = i - 1  # 更新索引 j
    ci = S / z  # 计算一致性指数
    return ci