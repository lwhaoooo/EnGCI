import torch
import torch.nn as nn

# ResNetBlock的定义保持不变
class ResNetBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.5):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.bn2(y)
        out = self.relu(x + y)
        return out

# Kolmogorov-Arnold Network (KAN) 定义
class KolmogorovArnoldNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(KolmogorovArnoldNetwork, self).__init__()
        self.lambdas = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.phi = nn.ReLU()  # 激活函数，选择 ReLU 作为示例
        self.output_weights = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 计算单变量函数组合
        hidden_output = self.phi(torch.matmul(x, self.lambdas.T))
        # 线性组合得到最终输出
        output = self.output_weights(hidden_output)
        return output

# AffinityModel的定义
class AffinityModel(nn.Module):
    def __init__(self):
        super(AffinityModel, self).__init__()
        
        # Drug feature network with 1D-CNN and ResNet blocks
        self.drug_conv = nn.Sequential(
            ResNetBlock(channels=512),
            ResNetBlock(channels=512)
        )

        # Protein feature network with additional convolutional layers and ResNet blocks
        self.protein_conv = nn.Sequential(
            ResNetBlock(channels=1280),
            ResNetBlock(channels=1280),
            nn.Conv1d(in_channels=1280, out_channels=640, kernel_size=8, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(640),
            nn.Conv1d(in_channels=640, out_channels=320, kernel_size=8, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(320),
            nn.Conv1d(in_channels=320, out_channels=128, kernel_size=8, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        # Reduce protein features to match the embedding dimension of the drug features
        self.reduce_protein_dim = nn.Linear(128, 512)

        # Bidirectional attention mechanism
        self.attention_drug_to_protein = nn.MultiheadAttention(embed_dim=512, num_heads=4)
        self.attention_protein_to_drug = nn.MultiheadAttention(embed_dim=512, num_heads=4)

        # 使用 KAN 替换原来的 MLP
        self.kan_combined = KolmogorovArnoldNetwork(input_dim=1024, hidden_dim=256, output_dim=1)
        
    def forward(self, drug_feature, protein_feature):
        # Reshape and process drug features
        drug_feature = drug_feature.unsqueeze(2)
        drug_feature = self.drug_conv(drug_feature)
        drug_feature = torch.mean(drug_feature, dim=2)  # Global average pooling

        # Process protein features
        protein_feature = protein_feature.unsqueeze(2)
        protein_feature = self.protein_conv(protein_feature)
        protein_feature = torch.mean(protein_feature, dim=2)  # Global average pooling

        # Reduce protein feature dimension to match drug feature dimension
        protein_feature = self.reduce_protein_dim(protein_feature)

        # Bidirectional attention
        drug_feature = drug_feature.unsqueeze(0)
        protein_feature = protein_feature.unsqueeze(0)
        
        attn_output_drug, _ = self.attention_drug_to_protein(drug_feature, protein_feature, protein_feature)
        attn_output_protein, _ = self.attention_protein_to_drug(protein_feature, drug_feature, drug_feature)
        
        # Combine the drug and protein features
        combined_features = torch.cat((attn_output_drug.squeeze(0), attn_output_protein.squeeze(0)), dim=1)

        # 使用 KAN 网络进行分类
        x = self.kan_combined(combined_features)
        
        return x