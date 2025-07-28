import torch
import torch.nn as nn
import torch.nn.functional as F
from methods import cluster_patch, enhance_normal, enhance_abnormal, enhance_data, enhance_seq
import random
import math
from mymodel import TSDE_base

class ContrastiveLearningModel(nn.Module):
    def __init__(self, config, device):
        super(ContrastiveLearningModel, self).__init__()
        # 定义一个简单的全连接网络作为特征提取器（encoder）
        target_dim = config["embedding"]["num_feat"]
        self.device = device
        self.encoder = TSDE_base(target_dim=target_dim, config=config, device=device, sample_feat=None)

    def forward(self, observed_data, feature_id=None):
        # 对输入进行特征提取，返回嵌入
        mts_emb, xt_proj, xf_proj =  self.encoder(observed_data, feature_id)
        return mts_emb, xt_proj, xf_proj
    

    def info_nce_loss(self, observed_data, positive_pairs, negative_pairs, feature_id=None,temperature=0.07):
        b, L, C = observed_data.shape
        # data = observed_data.transpose(-2, -1)  # 转换为 (batch_size, feature_size, win_size)
        # _,pos, indices = cluster_patch(data, 10, 2)
        
        # data_t = data.transpose( -2, -1)  # 转换为 (win_size, batch_size, feature_size)
        # enhanced_data = enhance_data(data_t, indices, 10)
        # enhanced_data = enhanced_data.to(self.device)
        
        # positive_pairs, negative_pairs = enhance_seq(enhanced_data, indices, 10)
        positive_pairs = [(a.to(self.device), b.to(self.device)) for (a, b) in positive_pairs]
        negative_pairs = [(a.to(self.device), b.to(self.device)) for (a, b) in negative_pairs]
        n = len(positive_pairs) + len(negative_pairs)
        # all_data是一个tensor， shape是(2*n, L, C)
        all_data = torch.zeros((2*n, L,C), device=self.device)
        # 填充正样本对
        for i, (a, b) in enumerate(positive_pairs):
            all_data[i*2] = a
            all_data[i*2 + 1] = b
        # 填充负样本对
        for i, (a, b) in enumerate(negative_pairs):
            all_data[len(positive_pairs)*2 + i*2] = a
            all_data[len(positive_pairs)*2 + i*2 + 1] = b
        
        """
        InfoNCE损失函数
        positive_pairs: [(a1, b1), (a2, b2), ...] 正样本对
        negative_pairs: [(a1, b1), (a2, b2), ...] 负样本对
        temperature: 缩放因子，通常为0.07
        """
        pos_similarities = []
        neg_similarities = []
        
        # 把 feature_id (batch_size, feature_size) 复制为 (len(all_data), feature_size)
        if feature_id is not None:
            feature_id = feature_id[0].expand(len(all_data), -1).to(self.device).long()  # 确保 feature_id 在正确的设备上
            # print(f"feature_id shape: {feature_id.shape}")
        # all_data_embed : B, 2*C, K, L, all_xt_proj, all_xf_proj : B, C, K, L
        all_data_embed, all_data_xt_proj, all_dataxf_proj = self.forward(all_data, feature_id)  # 获取所有数据的嵌入表示
        # B, 2*C, K, L 
        # print(f"all_data_embed shape: {all_data_embed.shape}")
        # 将嵌入表示分成正样本对和负样本对
        pos_embed = all_data_embed[:len(positive_pairs) * 2,:,:,:]
        neg_embed = all_data_embed[len(positive_pairs) * 2:,:,:]

        # print(f"pos_embed shape: {pos_embed.shape}, neg_embed shape: {neg_embed.shape}")
        # 把pos_embed和neg_embed分成正样本对和负样本对 pos_embed[0] :(a,b)
        nn,E, L, C = pos_embed.shape
        n = nn // 2  # 每个正样本对有两个样本
        pos_embed = pos_embed.view(n, 2,E, L, C)
        nn,E, L, C = neg_embed.shape
        n = nn // 2  # 每个负样本对有两个样本
        neg_embed = neg_embed.view(n, 2,E, L, C)

        # print(f"pos_embed shape: {pos_embed.shape}, neg_embed shape: {neg_embed.shape}")
        # 计算正样本对之间的相似度（余弦相似度）
        for i in range(len(pos_embed)):
            a_embed, b_embed = positive_pairs[i][0], positive_pairs[i][1]

            # 计算余弦相似度
            similarity = F.cosine_similarity(a_embed, b_embed)
            pos_similarities.append(similarity)
        
        # 计算负样本对之间的相似度
        for i in range(len(neg_embed)):
            a_embed, b_embed = neg_embed[i][0], neg_embed[i][1]

            # 计算余弦相似度
            similarity = F.cosine_similarity(a_embed, b_embed)
            neg_similarities.append(similarity)
        
        # 将正负样本对的相似度转换为张量
        pos_similarities = torch.stack(pos_similarities)  # 使用stack来生成张量
        neg_similarities = torch.stack(neg_similarities)  # 使用stack来生成张量
        
        # 正样本的相似度经过指数函数缩放
        pos_similarities_exp = torch.exp(pos_similarities / temperature).sum(dim=0)
        neg_similarities_exp = torch.exp(neg_similarities / temperature).sum(dim=0)
        # print(f"pos_similarities_exp: {pos_similarities_exp}")
        # print(f"neg_similarities_exp: {neg_similarities_exp}")
        # print(f"pos_similarities_exp shape: {pos_similarities_exp.shape}")
        # print(f"neg_similarities_exp shape: {neg_similarities_exp.shape}")

        
        # 总的相似度分母，正样本和负样本的权重计算
        total_similarity = pos_similarities_exp + neg_similarities_exp
        
        # InfoNCE损失
        loss = -torch.log(pos_similarities_exp / total_similarity).mean()  # 平均损失
        return loss, pos_embed, neg_embed

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
    observed_data = torch.randn(8, 100,25)  # 假设输入数据为(batch_size, feature_size, win_size)
    # data = torch.randn(32, 5, 100)  
    # _,pos, indices = cluster_patch(data, 10, 2)
    # print(f"indices shape: {indices.shape}")
    # data_t = data.transpose( -2, -1)  # 转换为 (win_size, batch_size, feature_size)
    # enhance_data = enhance_data(data_t, indices, 10)
    # print(f"enhance_data: {enhance_data}")
    # print(f"enhance_data shape: {enhance_data.shape}")
    # positive_pairs, negative_pairs = enhance_seq(enhance_data, indices, 10)
    # print(f"positive_pairs.shape: {positive_pairs[0][1].shape}")


    model = ContrastiveLearningModel(config, device='cuda:0')
    loss = model.info_nce_loss(observed_data)
    print(f"Contrastive loss: {loss.item()}")
