import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import anomaly_detection_dataloader

from mtsEmb import embedding_MTS

class TSDE_base(nn.Module):
    
    def __init__(self, target_dim, config, device, sample_feat):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.sample_feat = sample_feat

        self.emb_time_dim = config["embedding"]["timeemb"]
        self.emb_cat_feature_dim = config["embedding"]["featureemb"]
        self.mts_emb_dim = 1 + 2 * config["embedding"]["channels"]
        
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_cat_feature_dim
        )
        self.embed_layer = self.embed_layer.to(self.device)

        config_emb = config["embedding"]
        self.embdmodel = embedding_MTS(config_emb).to(self.device)

        L = config_emb["num_timestamps"]
        K = config_emb["num_feat"]
        num_classes = config_emb["classes"]

        self.mlp = nn.Sequential(
            nn.Linear(L * K * self.mts_emb_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self.conv = nn.Linear((self.mts_emb_dim - 1) * K, K, bias=True).to(self.device)

    def time_embedding(self, pos, d_model=128):
        # print("pos shape:", pos.shape, "d_model:", d_model)
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model)
        # print("pe shape:", pe.shape, "position shape:", position.shape, "div_term shape:", div_term.shape)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_mts_emb(self, observed_data, feature_id):
        # print(f"observed_data shape: {observed_data.shape}, feature_id: {feature_id}")
        data = observed_data.transpose(-2, -1)  # 转换为 (batch_size, feature_size, win_size)
        B, K, L = data.shape
        data = data.to(self.device).float()  # 转换为 (batch_size, feature_size, win_size)
        x_co = data.unsqueeze(1) 
        tp = np.tile(np.arange(L), (B, 1)) * 1.0
        observed_tp = torch.tensor(tp).to(self.device).float()  # 确保它在正确的设备上
        
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)

        if feature_id is None:
            feature_embed = self.embed_layer(torch.arange(self.target_dim).to(self.device))  # 确保在正确的设备上
            feature_embed = feature_embed.unsqueeze(0).expand(B, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id.to(self.device).long())  # 确保 feature_id 在正确的设备上
            feature_embed = feature_embed.to(self.device)

        cond_embed, xt, xf = self.embdmodel(x_co.to(self.device), time_embed.to(self.device), feature_embed.to(self.device))

        mts_emb = cond_embed
        # print(mts_emb.shape)
        return mts_emb, xt, xf 

    def forward(self, observed_data, feature_id=None):
        observed_data = observed_data.to(self.device).float()  # 转换为 (batch_size, feature_size, win_size)
        
        mts_emb, xt_proj, xf_proj = self.get_mts_emb(observed_data, feature_id)
        
        # print(mts_emb.shape)
        return mts_emb, xt_proj, xf_proj

# 测试代码
if __name__ == "__main__":
    config = {
        "embedding": {
            "timeemb": 128,
            "featureemb": 16,
            "num_feat": 25,
            "num_timestamps": 100,
            "classes": 2,
            "channels": 16,
            "nheads": 8
        }
    }
    
    # model = TSDE_base(target_dim=25, config=config, device='cuda:0', sample_feat=None)
    
    # observed_data = torch.randn(32, 100, 25)  # 假设输入数据为(batch_size, feature_size, win_size)
    # feature_id = None
    # output = model(observed_data, feature_id)
    # print(output)

    train_loader, valid_loader, test_loader = anomaly_detection_dataloader(dataset_name='PSM', batch_size=32)

    for batch in train_loader:
        observed_data, feature_id = batch["observed_data"], batch["feature_id"]
        observed_data = observed_data.to('cuda:0')
        feature_id = feature_id.to('cuda:0') if feature_id is not None else None
        model = TSDE_base(target_dim=25, config=config, device='cuda:0', sample_feat=None)
        output = model(observed_data, feature_id)

        # print(output)
        break