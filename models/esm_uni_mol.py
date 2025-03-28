import torch
import torch.nn as nn
import pdb

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
            ResNetBlock(channels=1280)
        )

        # Reduce protein features to match the embedding dimension of the drug features
        self.reduce_protein_dim = nn.Linear(1280, 512)

        # Bidirectional attention mechanism
        self.attention_drug_to_protein = nn.MultiheadAttention(embed_dim=512, num_heads=4)
        self.attention_protein_to_drug = nn.MultiheadAttention(embed_dim=512, num_heads=4)

        self.kan_combined = KANLinear(in_features=1024, out_features=1, grid_size=5, spline_order=3, 
                                      scale_noise=0.1, scale_base=1.0, scale_spline=1.0, 
                                      enable_standalone_scale_spline=True, base_activation=nn.SiLU)

        # self.fc_combined = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(512),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 1)
        # )
        
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

        # Combine the drug and protein features
        combined_features = torch.cat((drug_feature, protein_feature), dim=1)

        x = self.kan_combined(combined_features)
        
        return x
