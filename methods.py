import os
import numpy as np
import torch.nn.functional as F
import random
import torch
import torch.nn as nn
import pandas as pd
from sklearn.cluster import KMeans as Kmeans
# dataloader 在本文件的上一级目录
from dataloader import SWATSegLoader, SMAPSegLoader, SMDSegLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import warnings
warnings.filterwarnings("ignore")
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

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
    indices = torch.zeros(batch_size, num_patches, dtype=torch.long)
    for i in range(patches.shape[0]):
        cluster_centers = []
        
        patches_i = patches[i].reshape(num_patches,-1) 
        # print(f"patches_i shape: {patches_i.shape}")
        patches_i = patches_i.cpu().numpy()
        kmeans = Kmeans(n_clusters=cluster_num,  init='random', random_state=42)
        kmeans.fit(patches_i)        
        cluster_centers.append(kmeans.cluster_centers_)
        distances = kmeans.transform(patches_i)  # distances的shape为(num_patches, cluster_num)
        distances = distances.sum(axis=1, keepdims=True)
        distances = distances.squeeze()
        farthest_indices = torch.topk(torch.tensor(distances), 3, largest=True).indices
        indices[i,farthest_indices] = 1
        # print(f"farthest_indices: {farthest_indices}")
        for idx in farthest_indices:
            abnormaly_pos[i, :, idx * patch_size:(idx + 1) * patch_size] = 1
            # print(f"abnormaly_pos[i, :, {idx * patch_size}:{(idx + 1) * patch_size}]: {abnormaly_pos[i, :, idx * patch_size:(idx + 1) * patch_size]}")
    
    # print(f"abnormaly_pos shape: {abnormaly_pos.shape}")
    # print(f"abnormaly_pos: {abnormaly_pos}")
    
    return cluster_centers, abnormaly_pos, indices

def enhance_normal(patch):
    # 四种数据增强方法，随机选择一种进行
    # 自定义一个数，比如0-3，随机选择一个数，
    method = random.randint(0, 3)
    method = 2
    methods= ["random_crop", "flip", "rotate", "noise"]
    # print(f"Enhancing patch with method {methods[method]}")
    
    if method == 0:
        # 1. 随机裁剪, patch[:] 变成patch[x:]+patch[:x]
        n = patch.shape[0]
        x = random.randint(0, n - 1)
        patch = torch.cat((patch[x:], patch[:x]), dim=0)
    elif method == 1:
        # 2. 随机翻转
        patch = torch.flip(patch, dims=[0])
    elif method == 2:
        # 3. 随机旋转  把其中一段patch旋转180度
        x = random.randint(0, patch.shape[0] - 1)
        y = random.randint(0, patch.shape[0] - 1)
        # print(f"Rotating patch from {x} to {y}")
        if x > y:
            patch = torch.cat((patch[:x], torch.flip(patch[x:x+y], dims=[0]), patch[x+y:]), dim=0)
        else:
            patch = torch.cat((patch[:y], torch.flip(patch[y:x+y], dims=[0]), patch[x+y:]), dim=0)
        # print(f"patch shape after rotation: {patch.shape}")
    elif method == 3:
        # 4. 随机加噪
        noise = torch.randn_like(patch) * 0.01
        patch = patch + noise
    return patch

def enhance_abnormal(patch): 
    # 两种增强方法，随机选择一种进行
    method = random.randint(0, 1)
    methods = ["linear", "noise"]
    if method == 0:
        # 线性变换
        scale = random.uniform(0.5, 2)
        beta = random.uniform(-0.1, 0.1)
        patch = patch * scale + beta
    elif method == 1:
        # 加噪声
        noise = torch.randn_like(patch) * 0.5
        patch = patch + noise
    return patch

def enhance_data(data, indices, patch_size):    
    # data : (bs, win_size, feature_size)
    # indices : (bs, num_patches)  # 1表示该patch是异常的, 0表示该patch是正常的

    batch_size, win_size, feature_size = data.shape
    num_patches = indices.shape[1]
    enhanced_patches = torch.zeros(batch_size, num_patches, patch_size, feature_size)
    for i in range(batch_size):
        for j in range(num_patches):
            patch = data[i, j * patch_size:(j + 1) * patch_size, :]
            if indices[i, j] == 1:  # 异常patch
                enhanced_patch = enhance_abnormal(patch)
                enhanced_patches[i][j] = enhanced_patch
            else:  # 正常patch
                enhanced_patch = enhance_normal(patch)
                enhanced_patches[i][j] = enhanced_patch
    enhanced_data = enhanced_patches.view(batch_size, num_patches * patch_size, feature_size)
    return enhanced_data

def enhance_seq(data, indices, patch_size, n = 1):
    original_data = data.clone()  # 保留原始数据
    # data : (bs, win_size, feature_size)
    # indices : (bs, num_patches)  # 1表示该patch是异常的, 0表示该patch是正常的

    # 对于每一个样本， 其shape为(win_size, feature_size)， 随机选择其中的一个patch进行增强，在随机选择另外一个样本的patch， 两个patch进行交换
    batch_size, win_size, feature_size = data.shape
    num_patches = indices.shape[1]   
    enhanced_data = torch.zeros_like(data)
    positive_pairs = []
    negative_pairs = []
    for i in range(batch_size):
        random_numbers = random.sample(range(0, num_patches), n)
        # print("random_numbers:", random_numbers)
        for j in random_numbers:
            patch = data[i, j * patch_size:(j + 1) * patch_size, :]
            # random_sample = random.randint(0, batch_size - 1)
            # random_patch_index = random.randint(0, num_patches - 1)
            # random_patch = data[random_sample, random_patch_index * patch_size:(random_patch_index + 1) * patch_size, :]
            find_positive = False
            find_negative = False
            while not find_positive or not find_negative:
                random_sample = random.randint(0, batch_size - 1)
                random_patch_index = random.randint(0, num_patches - 1)
                random_patch = data[random_sample, random_patch_index * patch_size:(random_patch_index + 1) * patch_size, :]
                if indices[i, j] == indices[random_sample, random_patch_index] and not find_positive:  # 两个patch都是异常的，增强后仍是异常的， 两个patch都是正常的，增强后仍是正常的
                    # 两个patch都是异常的，增强后仍是异常的， 两个patch都是正常的，增强后仍是正常的
                    # print("positive pair is found")
                    find_positive = True
                    data[i, j * patch_size:(j + 1) * patch_size, :] = random_patch
                    x = torch.cat([data[i,:patch_size*j,:], random_patch, data[i, (j + 1) * patch_size:, :]], dim=0)
                    positive_pairs.append((x, original_data[i]))
                    y = torch.cat([data[random_sample, :random_patch_index * patch_size, :], patch, data[random_sample, (random_patch_index + 1) * patch_size:, :]], dim=0)
                    positive_pairs.append((y, original_data[random_sample]))
                    # print(f"positive_pairs: {len(positive_pairs)}")
                elif indices[i, j] != indices[random_sample, random_patch_index] and not find_negative:  # 两个patch都是异常的，增强后仍是异常的， 两个patch都是正常的，增强后仍是正常的
                    # 两个patch都是异常的，增强后仍是异常的， 两个patch都是正常的，增强后仍是正常的
                    # print("negative pair is found")
                    find_negative = True
                    data[i, j * patch_size:(j + 1) * patch_size, :] = random_patch
                    x = torch.cat([data[i,:patch_size*j,:], random_patch, data[i, (j + 1) * patch_size:, :]], dim=0)
                    negative_pairs.append((x, original_data[i]))
                    y = torch.cat([data[random_sample, :random_patch_index * patch_size, :], patch, data[random_sample, (random_patch_index + 1) * patch_size:, :]], dim=0)
                    negative_pairs.append((y, original_data[random_sample]))
                    # print(f"negative_pairs: {len(negative_pairs)}")
    # print(f"positive_pairs: {len(positive_pairs)}, negative_pairs: {len(negative_pairs)}")
    # print(f"positive_pairs[0]: {positive_pairs[0][0].shape}, negative_pairs[0]: {negative_pairs[0][0].shape}")
    return positive_pairs, negative_pairs


def enhance_seq111(data, indices, patch_size, n = 2):
    original_data = data.clone()  # 保留原始数据
    # data : (bs, win_size, feature_size)
    # indices : (bs, num_patches)  # 1表示该patch是异常的, 0表示该patch是正常的

    # 对于每一个样本， 其shape为(win_size, feature_size)， 随机选择其中的一个patch进行增强，在随机选择另外一个样本的patch， 两个patch进行交换
    batch_size, win_size, feature_size = data.shape
    num_patches = indices.shape[1]   
    enhanced_data = torch.zeros_like(data)
    positive_pairs = []
    negative_pairs = []
    for i in range(batch_size):
        random_numbers = random.sample(range(0, num_patches), 1)
        for j in random_numbers:
            patch = data[i, j * patch_size:(j + 1) * patch_size, :]
            random_sample = random.randint(0, batch_size - 1)
            random_patch_index = random.randint(0, num_patches - 1)
            random_patch = data[random_sample, random_patch_index * patch_size:(random_patch_index + 1) * patch_size, :]
            # 第i个样本的第j个patch与随机选择的样本的random_patch_index个patch进行增强
            if indices[i, j] ==indices[random_sample, random_patch_index]:  # 两个patch都是异常的，增强后仍是异常的， 两个patch都是正常的，增强后仍是正常的
                    # 正样本对
                data[i, j * patch_size:(j + 1) * patch_size, :] = random_patch
                x = torch.cat([data[i,:patch_size*j,:], random_patch, data[i, (j + 1) * patch_size:, :]], dim=0)
                positive_pairs.append((x, original_data[i]))
                y = torch.cat([data[random_sample, :random_patch_index * patch_size, :], patch, data[random_sample, (random_patch_index + 1) * patch_size:, :]], dim=0)
                positive_pairs.append((y, original_data[random_sample]))

            else:
                data[i, j * patch_size:(j + 1) * patch_size, :] = random_patch
                x = torch.cat([data[i,:patch_size*j,:], random_patch, data[i, (j + 1) * patch_size:, :]], dim=0)
                negative_pairs.append((x, original_data[i]))
                y = torch.cat([data[random_sample, :random_patch_index * patch_size, :], patch, data[random_sample, (random_patch_index + 1) * patch_size:, :]], dim=0)
                negative_pairs.append((y, original_data[random_sample]))
    # print(f"positive_pairs: {len(positive_pairs)}, negative_pairs: {len(negative_pairs)}")
    return positive_pairs, negative_pairs

def train(model, train_loader, test_loader, optimizer, device, foldername, epochs=1, normalize="True"):

    model.train()
    total_loss = 0
    num_epochs = epochs
    p1 = int(0.75 * num_epochs )
    p2 = int(0.9 * num_epochs )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    if foldername != "":
        output_path = foldername + "model.pth"
        loss_path = foldername + "losses.txt"
    
    for epoch_no in range(1, num_epochs + 1):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss, _ = model(train_batch["observed_data"].to(device), train_batch["feature_id"],device=device, normalize =normalize)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                
            
            ## Save Losses in txt File
            if foldername != "":
                ## Save Losses in txt File
                with open(loss_path, "a") as file:
                    file.write('avg_epoch_loss: '+ str(avg_loss / batch_no) + ", epoch= "+ str(epoch_no) + "\n")
            lr_scheduler.step()
    if foldername != "":
        torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")
    print(f"Training complete. Average loss: {avg_loss / batch_no}")

    test(model, train_loader, test_loader, foldername, device)


def test(model, train_loader,test_loader, result_folder, device, anomaly_ratio=1, normalize="True"):
    attens_energy = []  # energy of the test data, just reconstruction error
    train_energies = []  # energy of the train data, just reconstruction error
    test_labels = []
    

    with torch.no_grad():
        model.eval()



        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            # test data evaluation
            for batch_no, test_batch in enumerate(it, start=1):
                # outputs: samples  result : MSE between samples and target
                loss, result = model.evaluate(test_batch["observed_data"].to(device), test_batch["feature_id"],device=device)

                attens_energy.append(result)
                test_labels.append(test_batch["label"])
                it.set_postfix(
                    ordered_dict={
                        "mse:": loss.item(),
                        "batch_no": batch_no,
                    },
                    refresh=False,
                )
                
                

        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, batch in enumerate(it, start=1):
                # reconstruction
                loss, result = model.evaluate(batch["observed_data"].to(device), batch["feature_id"],device=device)
                train_energies.append(result)
                # print('Output shape', outputs.shape)
                # print('Score shape', score.shape)
        print("train_energies:", len(train_energies), "train_energies[0]:", train_energies[0].shape)
        print("attens_energy:", len(attens_energy), "attens_energy[0]:", attens_energy[0].shape)
        # train_energies = np.concatenate([t.cpu().numpy() for t in train_energies], axis=0).reshape(-1)
        
        # train_energy = np.array(train_energies)
        train_energy = np.concatenate(train_energies, axis=0).reshape(-1)
        print("train_energy:", train_energy.shape)
        # (2) find the threshold
        # attens_energy = np.concatenate([t.cpu().numpy() for t in attens_energy], axis=0).reshape(-1)
        # attens_energy = np.array(attens_energy)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        print("attens_energy:", attens_energy.shape)
        test_energy = np.array(attens_energy)
        print("test_energy:", test_energy.shape)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        print("combined_energy:", combined_energy.shape)

        threshold = np.percentile(combined_energy, 100 - 1)
        threshold_2 = 0.5
        # threshold = 1
        # threshold = 8
        print("anomaly_ratio :", anomaly_ratio)
        print("Threshold :", threshold)
        print("attens_energy :", attens_energy.shape)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        print("########## not use detection adjustment ############")

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary',
                                                                                zero_division=1)

        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        ### 保存pred 和 gt


        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)
        

        print("pred and gt have been saved as .npy files in 'folder' directory.")

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary',
                                                                                zero_division=1)
    
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        with open(result_folder + 'results.txt', "a") as file:
            file.write("threshold: {:0.4f}\n".format(threshold))
            file.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))

            
        
        
    

if __name__ == "__main__":
    # 测试数据
    # data = torch.randn(2, 15, 100)  # 1个样本，15个特征，100个时间点
    # patch_size = 10
    # cluster_num = 2
    # cluster_center = cluster_patch(data, patch_size, cluster_num)
    # data = SWATSegLoader(100, 100, 'train')
    # patch = data[1]
    # observed_data = patch['observed_data'] # shape (win_size, feature_size)
    # # observed_data 是numpy， 转成tensor
    # observed_data = pd.DataFrame(observed_data)  # 转换为DataFrame
    # observed_data = torch.tensor(observed_data.values, dtype=torch.float32)  # 转换为tensor
    # observed_data = observed_data.permute(1, 0).unsqueeze(0) # shape (1, feature_size, win_size)
    # print(f"observed_data shape: {observed_data.shape}")
    # _, pos, farthest_indices = cluster_patch(observed_data, 10, 2)
    # print("farthest_indices:", farthest_indices)



    # 测试增强函数
    # patch = torch.randn(10, 5)  # 假设patch的shape为(10, 5)
    # enhanced_patch = enhance(patch)
    # print("Original patch:", patch)
    # print("original patch shape:", patch.shape)
    # print("Enhanced patch:", enhanced_patch)
    # print("enhanced patch shape:", enhanced_patch.shape)

    # 测试数据增强
    data = torch.randn(2, 5, 100)  
    _,pos, indices = cluster_patch(data, 10, 2)
    print(f"indices shape: {indices.shape}")
    data_t = data.transpose( -2, -1)  # 转换为 (win_size, batch_size, feature_size)
    enhance_data = enhance_data(data_t, indices, 10)
    print(f"enhance_data: {enhance_data}")
    print(f"enhance_data shape: {enhance_data.shape}")

    enhance_seq(enhance_data, indices, 10)
    

