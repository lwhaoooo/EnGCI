import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import esm
import os

file_path_train = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/GPCR_train.csv"
file_path_test = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/GPCR_test.csv"

# 读取数据
data = pd.read_csv(file_path_test)

# 蛋白质序列去重
data_cleaned_protein = data.drop_duplicates(subset=['target_sequence'])

# 转化为列表
compound_protein = list(data_cleaned_protein['target_sequence'])

# 将列表转换为字典，键和值都为列表中的值
compound_protein_dict = {idx: value for idx, value in enumerate(compound_protein)}

# 将字典转换为元组列表
compound_protein_list_tuple = list(compound_protein_dict.items())

# 加载 ESM-2 预训练模型和字母表
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

# 从 alphabet 对象中获取一个批量转换器（batch converter），该转换器用于将一组蛋白质序列转换成模型可以处理的格式。
batch_converter = alphabet.get_batch_converter()

# 确保模型处于评估模式
model.eval()

# 定义 Dataset 类
class ProteinDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

# 创建 Dataset 和 DataLoader
dataset = ProteinDataset(compound_protein_list_tuple)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=batch_converter)  # 将 batch_size 减小

# 处理批量数据并保存到磁盘
def process_in_batches_and_save(dataloader, model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for batch_idx, (batch_labels, batch_strs, batch_tokens) in enumerate(dataloader):
            result = model(batch_tokens, repr_layers=[33], return_contacts=True)
            
            # 保存结果到磁盘
            torch.save(result, os.path.join(output_dir, f"batch_{batch_idx}.pt"))
            
            del batch_tokens, result
            torch.cuda.empty_cache()  # 如果使用GPU，清理缓存

# 设置输出目录
output_dir = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/processed/esm_test"

# 处理数据并保存结果
process_in_batches_and_save(dataloader, model, output_dir)

print("Processing and saving completed.")
