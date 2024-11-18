import torch
import os
import re
import pickle
import pandas as pd
import esm
from tqdm import tqdm
import pdb

# 设置保存结果的目录
output_dir_train = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/processed/esm_train"
output_dir_test = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/processed/esm_test"

file_path_train = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/GPCR_train.csv"
file_path_test = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/GPCR_test.csv"

# 读取数据
data = pd.read_csv(file_path_test)

# 蛋白质序列去重
data_cleaned_protein = data.drop_duplicates(subset=['target_sequence'])

# 转化为列表
compound_smiles = list(data_cleaned_protein['target_sequence'])

# 将列表转换为字典，键和值都为列表中的值
compound_smiles_dict = {idx: value for idx, value in enumerate(compound_smiles)}

# 将字典转换为元组列表
compound_smiles_list_tuple = list(compound_smiles_dict.items())

# 加载 ESM-2 预训练模型和字母表
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

# 从 alphabet 对象中获取一个批量转换器（batch converter），该转换器用于将一组蛋白质序列转换成模型可以处理的格式。
batch_converter = alphabet.get_batch_converter()

# 使用批量转换器将数据转换成模型输入格式
batch_labels, batch_strs, batch_tokens = batch_converter(compound_smiles_list_tuple)

batch_lens = (batch_tokens != alphabet.padding_idx).sum(1).tolist()

protein_sequence = [batch_strs[i:i + 2] for i in range(0, len(batch_strs), 2)]
batch_lens_sequence = [batch_lens[i:i + 2] for i in range(0, len(batch_lens), 2)]

# 自然排序函数
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

# 定义逐批加载和处理结果的函数
def process_saved_results(output_dir_test):
    sequence_feature_dict = {}
    file_index = 0  # 用于跟踪处理到哪个文件了
    for filename in sorted(os.listdir(output_dir_test), key=natural_sort_key):
        if filename.endswith(".pt"):
            filepath = os.path.join(output_dir_test, filename)
            result = torch.load(filepath)
            token_representations = result['representations'][33]
            
            for i, sequence in enumerate(protein_sequence[file_index]):
                # pdb.set_trace()
                tokens_len = batch_lens_sequence[file_index][i]
                feature_vector = token_representations[i, 1:tokens_len - 1].mean(0)  # 对有效长度进行归一化
                sequence_feature_dict[sequence] = feature_vector

            file_index += 1
            del result
            torch.cuda.empty_cache()  # 清理缓存以防内存爆炸
    return sequence_feature_dict

# 调用逐批处理函数并获取字典
sequence_feature_dict = process_saved_results(output_dir_test)

# 打印字典中的部分内容以进行检查
for seq, feature in list(sequence_feature_dict.items())[:5]:
    print(f"Sequence: {seq}, Feature vector shape: {feature.shape}")

# # 如果需要保存字典到文件，可以使用如下代码

# 指定保存文件的目录和文件名
save_dir = "/home/fox/Biomedical_science/GraphDTA/GraphDTA-master/data/processed/esm_dict"
save_filename = "sequence_feature_dict_test.pkl"

save_path = os.path.join(save_dir, save_filename)

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# 保存字典到文件
with open(save_path, "wb") as f:
    pickle.dump(sequence_feature_dict, f)

print(f"Dictionary saved to {save_path}")