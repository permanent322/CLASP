import numpy as np
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import pandas as pd
from sklearn.cluster import KMeans as Kmeans

def cluster_patch(data, patch_size, cluster_num):
    '''
    将数据划分为多个patch，并对每个patch进行聚类
    eg: 把data分为10个patch, 对着10个patch进行聚类，得到2个聚类中心
        计算每个patch到每个聚类中心的距离， 找到距离最远的3个patch
    data: 输入数据，shape为(batch_size, feature_size, win_size)
    patch_size: 每个patch的大小
    cluster_num: 聚类的中心数量
    '''
    batch_size, feature_size, win_size = data.shape
    num_patches = win_size // patch_size
    patches = data.view(batch_size, feature_size, num_patches, patch_size)
    patches = patches.permute(0, 2, 1, 3) # 将patches的维度调整为(batch_size, num_patches, feature_size, patch_size)
    abnormaly_pos = torch.zeros_like(data, dtype=torch.float32)
    for i in range(patches.shape[0]):
        cluster_centers = []
        # pathches[i]的shape为(num_patches, feature_size, patch_size)
        # num_patches个patch一起聚类，得到cluster_num个聚类中心
        patches_i = patches[i].reshape(num_patches,-1) 
        print(f"patches_i shape: {patches_i.shape}")
        # 使用KMeans进行聚类 kmeans的输入是一个二维数组，行数为num_patches，列数为feature_size * patch_size
        kmeans = Kmeans(n_clusters=cluster_num, random_state=0)
        kmeans.fit(patches_i)        
        cluster_centers.append(kmeans.cluster_centers_)
        # 计算每个patch到每个聚类中心的距离
        distances = kmeans.transform(patches_i)  # distances的shape为(num_patches, cluster_num)
        # print(f"distances shape: {distances.shape}")
        # distances变为(num_patches, 1) 每个patch到所以聚类中心的距离的和
        distances = distances.sum(axis=1, keepdims=True)
        
        # print(f"distances shape: {distances.shape}")
        # print(f"distances: {distances}")
        distances = distances.squeeze()
        # print(f"distances after squeeze: {distances.shape}")
        # 找到距离最远的3个patch的索引
        farthest_indices = torch.topk(torch.tensor(distances), 3, largest=True).indices
        # print(f"farthest_indices: {farthest_indices}")
        # 将距离最远的3个patch的索引设置为1，其余设置为0
        print(f"farthest_indices: {farthest_indices}")
        # 将abnormaly_pos[i]的对应位置设置为1
        for idx in farthest_indices:
            print(idx)
            # idx是patch的索引，patch的shape为(num_patches, feature_size, patch_size)
            # 将对应的patch位置设置为1
            abnormaly_pos[i, :, idx * patch_size:(idx + 1) * patch_size] = 1
            print(f"abnormaly_pos[i, :, {idx * patch_size}:{(idx + 1) * patch_size}]: {abnormaly_pos[i, :, idx * patch_size:(idx + 1) * patch_size]}")
    # 将abnormaly_pos的shape调整为(batch_size, feature_size, win_size)
    #abnormaly_pos = abnormaly_pos.permute(0, 2, 1).reshape(batch_size, feature_size, win_size)
    print(f"abnormaly_pos shape: {abnormaly_pos.shape}")
    print(f"abnormaly_pos: {abnormaly_pos}")
    

    return cluster_centers, # abnormaly_pos

if __name__ == "__main__":
    # 测试数据
    data = torch.randn(2, 15, 100)  # 1个样本，15个特征，100个时间点
    patch_size = 10
    cluster_num = 2
    cluster_center = cluster_patch(data, patch_size, cluster_num)
