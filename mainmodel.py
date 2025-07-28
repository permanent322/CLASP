import torch
import torch.nn as nn
import torch.nn.functional as F
from CL import ContrastiveLearningModel
from methods import cluster_patch, enhance_normal, enhance_abnormal, enhance_data, enhance_seq

class Encoder(nn.Module):
    def __init__(self, config, device):
        super(Encoder, self).__init__()
        self.device = device
        target_dim = config["embedding"]["num_feat"]
        self.c = config["embedding"]["channels"]
        self.cl  = ContrastiveLearningModel(config=config, device=device)
    def forward(self, original_data, feature_id=None):
        b, L, K = original_data.shape
        data = original_data.transpose(-2, -1)  # 转换为 (batch_size, feature_size, win_size)
        _, pos, indices = cluster_patch(data, 10, 1)
        
        data_t = data.transpose( -2, -1)  # 转换为 (win_size, batch_size, feature_size)
        enhanced_data = enhance_data(data_t, indices, 10)
        enhanced_data = enhanced_data.to(self.device)
        positive_pairs, negative_pairs = enhance_seq(enhanced_data, indices, 10)
        # print("positive_pairs num:", len(positive_pairs), "negative_pairs num:", len(negative_pairs))
        cl_loss, pos_embed, neg_embed = self.cl.info_nce_loss(original_data, positive_pairs, negative_pairs, feature_id=feature_id)
        # cl_loss if a number
        # pos_embed and neg_embed are tensors:  B, 2,, 2*C, K, L 

        pos_embed = pos_embed.transpose(-2, -1)  # 转换为 (batch_size, 2*C, L, K)
        neg_embed = neg_embed.transpose(-2, -1)  # 转换为 (batch_size, 2*C, L, K)
        # print(f"pos_embed shape: {pos_embed.shape}, neg_embed shape: {neg_embed.shape}")
        # positive_pairs, negative_pairs, pos_embed, neg_embed 的shape要修改成一致
        data_nums = len(positive_pairs)*2 + len(negative_pairs)*2
        all_data = torch.zeros((data_nums, L, K), device=self.device)
        all_data_embed = torch.zeros((data_nums, 2*self.c, L, K), device=self.device)

        for i, (a, b) in enumerate(positive_pairs):
            
            all_data[i*2] = a
            all_data[i*2 + 1] = b
            all_data_embed[i*2] = pos_embed[i][0]
            all_data_embed[i*2 + 1] = pos_embed[i][1]

        for i, (a, b) in enumerate(negative_pairs):
            all_data[len(positive_pairs)*2 + i*2] = a
            all_data[len(positive_pairs)*2+ i*2 + 1] = b
            all_data_embed[len(positive_pairs)*2 + i*2] = neg_embed[i][0]
            all_data_embed[len(positive_pairs)*2 + i*2 + 1] = neg_embed[i][1]
        # print(f"all_data shape: {all_data.shape}, all_data_embed shape: {all_data_embed.shape}")

        return cl_loss, all_data, all_data_embed



class Decoder(nn.Module):
    def __init__(self, config, device):  # 新增 n_layers 参数，表示卷积层数
        super(Decoder, self).__init__()
        self.device = device
        
        self.embedding_dim = config["embedding"]["channels"] * 2  # Temporal + Spatial embedding
        self.feature_size = config["embedding"]["num_feat"]
        self.seq_len = config["embedding"]["seq_len"]
        self.hidden_dim = config["embedding"]["hidden_dim"]
        self.n_layers = config["embedding"].get("n_layers", 3)  # 默认3层卷积
        # 用 ModuleList 动态创建多个卷积层
        self.conv_layers = nn.ModuleList()
        
        # 创建 n_layers 层卷积
        for i in range(self.n_layers):
            # 动态调整每一层的输入和输出通道数
            in_channels = self.embedding_dim * self.feature_size if i == 0 else self.hidden_dim
            out_channels = self.hidden_dim if i < self.n_layers - 1 else self.feature_size  # 最后一层输出为 feature_size
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))

        # 最后的全连接层
        self.fc = nn.Linear(self.feature_size * self.seq_len, self.feature_size * self.seq_len)

    def forward(self, all_data_embed):
        # print(f"all_data_embed shape: {all_data_embed.shape}")
        
        B, C, L, D = all_data_embed.shape
        x = all_data_embed  # 输入数据形状: (B, embedding_dim * feature_size, seq_len)
        
        # 将输入调整为 (B, embedding_dim * feature_size, seq_len)
        x = x.contiguous().view(B, C * D, L)
        
        # 逐层通过卷积层处理数据
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))  # 使用 ReLU 激活函数
        
        # 3. 展平卷积输出并通过全连接层进行进一步映射
        x = x.view(B, -1)  # 展平为 (B, feature_size * seq_len)
        x = F.relu(self.fc(x))
        
        # 4. 重新调整形状为 (B, seq_len, feature_size)
        x = x.view(B, self.seq_len, self.feature_size)

        return x

   

        

class MyModel(nn.Module):
    def __init__(self, config, device):
        super(MyModel, self).__init__()
        self.device = device
        
        # 创建 Encoder 和 Decoder
        self.encoder = Encoder(config, device)
        self.decoder = Decoder(config, device)
        self.alpha = config["embedding"].get("alpha", 0.5)  # 默认值为0.5
        self.beta = config["embedding"].get("beta", 0.5)    #
        self.cl = ContrastiveLearningModel(config=config, device=device)
    def forward(self, original_data, feature_id=None, device="cuda:0", normalize="True"):
        # print(f"original_data shape: {original_data.shape}")
        if normalize == "True":
            # print("Normalizing the data")
            ## Normalization from non-stationary Transformer
            means = original_data.mean(1, keepdim=True)  # 沿着时间维度求均值
            original_data = original_data-means  # 减去均值
            stdev = torch.sqrt(torch.var(original_data, dim=1, keepdim=True, unbiased=False) + 1e-5)  # 沿着时间维度求标准差
            original_data /= stdev  # 除以标准差

        # Encoder 进行对比学习
        cl_loss, all_data, all_data_embed = self.encoder(original_data, feature_id)
        
        # Decoder 用于重建数据
        reconstructed_data = self.decoder(all_data_embed)
        # print(f"reconstructed_data shape: {reconstructed_data.shape}")
        # 计算重建损失
        reconstruction_loss = F.mse_loss(reconstructed_data, all_data)
        loss = self.alpha * cl_loss + self.beta * reconstruction_loss
        return loss, reconstructed_data
        print(f"reconstruction_loss.shape: {reconstruction_loss.shape}")
        print(f"reconstruction_loss: {reconstruction_loss.item()}")

    def evaluate(self, data, feature_id=None, device= "cuda:0"):
        """
        Evaluate the model on the given data.
        :param data: Input data for evaluation.
        :param feature_id: Optional feature ID for specific feature processing.
        :return: Reconstructed data.
        """
        original_data = data.to(self.device)
        means = original_data.mean(1, keepdim=True)  # 沿着时间维度求均值
        original_data = original_data-means  # 减去均值
        stdev = torch.sqrt(torch.var(original_data, dim=1, keepdim=True, unbiased=False) + 1e-5)  # 沿着时间维度求标准差
        original_data /= stdev  # 除以标准差
        data_emb, xt_proj, xf_proj = self.cl.forward(original_data, feature_id=feature_id)
        # print("data_emb shape:", data_emb.shape)
        data_emb = data_emb.transpose(-2, -1)  # 转换为 (batch_size, feature_size, win_size)
        reconstructed_data = self.decoder(data_emb).to(self.device)

        loss = F.mse_loss(reconstructed_data, original_data)
        # print(f"Evaluation loss: {loss.item()}")
        # print(f"Reconstructed data shape: {reconstructed_data.shape}")
        criterion= nn.MSELoss(reduction='none')
        
        score = torch.mean(criterion(original_data, reconstructed_data), dim=2)
        # print(f"Score shape: {score.shape}")
        score = score.detach().cpu().numpy()
        
        return loss, score
        

        
if __name__ == "__main__":
    # test encoder
    config = {
            "embedding": {
            "timeemb": 128,
            "featureemb": 16,
            "num_feat": 5,
            "num_timestamps": 100,
            "classes": 2,
            "channels": 16,
            "nheads": 8,
            "seq_len": 100, 
            "alpha": 0.5,
            "beta": 0.5,
            "hidden_dim": 128
            }
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder = Encoder(config, device)
    original_data = torch.randn(8, 100, 5)  # 假设输入数据为(batch_size, feature_size, win_size)
    # encoder.to(device)
    # cl_loss, all_data, all_data_embed = encoder(original_data)
    # # cprint(f"cl_loss: {cl_loss}, \n all_data: {all_data}, \n all_data_embed: {all_data_embed}")

    # Decoder = Decoder(config, device)
    # Decoder.to(device)
    # Decoder(all_data_embed)
    model = MyModel(config, device)
    model.to(device)
    cl_loss, reconstructed_data = model(original_data)

    model.evaluate(original_data)




# class Decoder(nn.Module):
#     def __init__(self, config, device):
#         super(Decoder, self).__init__()
#         self.device = device
        
#         # 这里embedding_dim表示 feature_emb + time_emb
#         self.embedding_dim = config["embedding"]["channels"] * 2  # Temporal + Spatial embedding
#         self.feature_size = config["embedding"]["num_feat"]
#         self.seq_len = config["embedding"]["seq_len"]
#         self.hidden_dim = config["embedding"]["hidden_dim"]
        
#         # 使用卷积层来处理时序数据
#         self.conv1 = nn.Conv1d(self.embedding_dim * self.feature_size, self.hidden_dim, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(self.hidden_dim, self.feature_size, kernel_size=3, padding=1)
        
#         # 最后的全连接层映射回目标形状
#         self.fc = nn.Linear(self.feature_size * self.seq_len, self.feature_size * self.seq_len)

#     def forward(self, all_data_embed):
#         print(f"all_data shape: {all_data.shape}, all_data_embed shape: {all_data_embed.shape}")
#         # 输入的x形状: (B, embedding_dim * feature_size, seq_len)
#         x = all_data_embed  # 使用all_data_embed作为输入
#         B, C, L, D = x.shape
        
#         # 1. 将输入调整为 (B, embedding_dim * feature_size, seq_len)
#         x = x.view(B, C * D, L)  # 合并embedding_dim和feature_size
        
#         # 2. 使用卷积层提取时序数据的局部特征
#         x = F.relu(self.conv1(x))  # (B, hidden_dim, seq_len)
#         x = self.conv2(x)          # (B, feature_size, seq_len)
        
#         # 3. 展平卷积输出并通过全连接层进行进一步映射
#         x = x.view(B, -1)  # 展平为 (B, feature_size * seq_len)
#         x = F.relu(self.fc(x))
        
#         # 4. 重新调整形状为 (B, feature_size, seq_len)
#         x = x.view(B, self.feature_size, self.seq_len)
        
#         return x