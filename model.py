import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from methods import cluster_patch, enhance_data

class TSDE_AD(nn.Module):
    def __init__(self, data, hidden_dim, patch_size, cluster_num, target_dim, config, device ):
        super(TSDE_AD, self).__init__()
        self.data = data
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.cluster_num = cluster_num
        self.target_dim = target_dim
        self.config = config


    def forward(self, observed_data, observed_mask):
        residual = observed_data
        # 这里可以添加模型的前向传播逻辑
        # 首先把observed_data分成patches
        _, abnormaly_pos, farthest_indices = cluster_patch(observed_data, self.patch_size, self.cluster_num)
        enhanced_data =  enhance_data(observed_data, farthest_indices, self.patch_size) # 整个数据集的增强 normal和abnormal patch的增强方式不一样
        

        return farthest_indices
    



if __name__ == "__main__":
    # 测试代码
    data = torch.randn(3, 10, 100)  # 假设输入数据为(batch_size, feature_size, win_size)
    model = TSDE_AD(data, hidden_dim=64, patch_size=10, cluster_num=2, target_dim=10, config={}, device='cuda:0')
    observed_data = torch.randn(3, 10, 100)  # 假设输入数据为(batch_size, feature_size, win_size)
    observed_mask = torch.ones(3, 10, 100)  # 假设mask全为1
    output = model(observed_data, observed_mask)
    print(output)