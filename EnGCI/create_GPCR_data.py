import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
import pdb

def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smile}")
        c_size = mol.GetNumAtoms()
        features = []
        for atom in mol.GetAtoms():
            feature = atom_features(atom)
            features.append(feature / sum(feature))

        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])

        return c_size, features, edge_index
    except Exception as e:
        print(f"Error processing SMILES {smile}: {e}")
        return None, None, None

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict.get(ch, 0)  # 使用字典的get方法，找不到字符则返回0
    return x

# 处理新数据集
train_dataset_path = '/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/GPCR/GPCR_train.txt'
test_dataset_path = '/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/GPCR/GPCR_test.txt'

def load_data(file_path):
    data = pd.read_csv(file_path, delimiter=' ', header=None)
    drugs = data[0].tolist()
    prots = data[1].tolist()
    affinity = data[2].tolist()
    return drugs, prots, affinity

# 加载训练集和测试集
train_drugs, train_prots, train_affinity = load_data(train_dataset_path)
test_drugs, test_prots, test_affinity = load_data(test_dataset_path)

# 创建数据集文件
def create_dataset_file(drugs, prots, affinity, data_type):
    file_path = f'data/GPCR_{data_type}.csv'
    with open(file_path, 'w') as f:
        f.write('compound_iso_smiles,target_sequence,affinity\n')
        for i in range(len(drugs)):
            ls = []
            ls += [drugs[i]]
            ls += [prots[i]]
            ls += [affinity[i]]
            f.write(','.join(map(str, ls)) + '\n')
    return file_path

# 创建训练集和测试集文件
train_file_path = create_dataset_file(train_drugs, train_prots, train_affinity, 'train')
test_file_path = create_dataset_file(test_drugs, test_prots, test_affinity, 'test')

# 定义序列字典和最大序列长度
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

# 生成SMILES图
compound_iso_smiles = []
for dt_name in ['GPCR']:
    for opt in ['train', 'test']:
        df = pd.read_csv(f'data/{dt_name}_{opt}.csv')
        compound_iso_smiles += list(df['compound_iso_smiles'])

compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    c_size, features, edge_index = smile_to_graph(smile)
    if c_size is not None:
        smile_graph[smile] = (c_size, features, edge_index)

datasets = ['GPCR']
for dataset in datasets:
    for opt in ['train', 'test']:
        processed_data_file = f'data/processed/{dataset}_{opt}.pt'
        if not os.path.isfile(processed_data_file):
            df = pd.read_csv(f'data/{dataset}_{opt}.csv')
            drugs, prots, Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])
            XT = [seq_cat(t) for t in prots]
            drugs, prots, Y = np.asarray(drugs), np.asarray(XT), np.asarray(Y)

            print(f'preparing {dataset}_{opt}.pt in pytorch format!')
            data = TestbedDataset(root='data', dataset=f'{dataset}_{opt}', xd=drugs, xt=prots, y=Y, smile_graph=smile_graph)
            print(f'{processed_data_file} has been created')
        else:
            print(f'{processed_data_file} is already created')
