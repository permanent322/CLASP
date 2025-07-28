import torch
import pandas as pd
from sklearn.cluster import KMeans as Kmeans
from dataloader import anomaly_detection_dataloader
import argparse
import json
import yaml
import os
import sys
import random
import numpy as np
import torch.nn as nn
from methods import train, test
from mainmodel import MyModel
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
from methods import set_seed
torch.backends.cudnn.enabled = False
import warnings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description="TSDE-Anomaly Detection")
parser.add_argument("--config", type=str, default="psm.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--normalize", type= str, default="True", help="Normalize the data")
parser.add_argument("--dataset", type=str, default='PSM')
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--run", type=str, default='1')
parser.add_argument("--anomaly_ratio", type=float, default=1, help="Anomaly ratio")
parser.add_argument("--mode", type=str, default='train', help="Mode: train or test or both")
args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

# config["model"]["mix_masking_strategy"] = args.mix_masking_strategy
# config = {
#         "embedding": {
#             "timeemb": 128,
#             "featureemb": 16,
#             "num_feat": 25,
#             "num_timestamps": 100,
#             "classes": 2,
#             "channels": 16,
#             "nheads": 8
#         }
#     }
print(json.dumps(config, indent=4))

mode = args.mode
set_seed(args.seed)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

batch_size = config["train"]["batch_size"]
foldername = "./save/Anomaly_Detection/" + args.dataset + "/run_" + str(args.run) +"/"
train_loader, valid_loader, test_loader = anomaly_detection_dataloader(dataset_name = args.dataset, batch_size = 32)

epochs = config["train"]["epochs"]
# 把config保存到文件中
if not os.path.exists(foldername):
    os.makedirs(foldername)
config_path = os.path.join(foldername, "config.yaml")
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)

if mode == "train":
    print(f"Training {args.dataset} with anomaly ratio {args.anomaly_ratio} for {epochs} epochs")
    model = MyModel(config, device).to(args.device)
    anomaly_ratio = args.anomaly_ratio
    model = model.to(device) 
    lr = config["train"]["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(model, train_loader, test_loader, optimizer, device, foldername, epochs=epochs, normalize=args.normalize)

elif mode == "test":
    print(f"Testing {args.dataset} with anomaly ratio {args.anomaly_ratio}")
    modelpath = "./save/Anomaly_Detection/" + args.dataset +"/"+ args.modelfolder +"/model.pth"
    # resultpath = "./save/Anomaly_Detection/" + args.dataset + "/run_" + str(args.run) +"/"
    # 加载本地模型
    model = MyModel(config, device).to(args.device)
    model.load_state_dict(torch.load(modelpath, map_location=device))
    model.eval()
    print(f"Model loaded from {modelpath}")
    test(model, train_loader, test_loader, foldername, device, anomaly_ratio=args.anomaly_ratio, normalize=args.normalize)
    # 进行测试




# model = ContrastiveLearningModel(config, device).to(args.device)
# train_loader, valid_loader, test_loader = anomaly_detection_dataloader(dataset_name = args.dataset, batch_size = 16)
# anomaly_ratio = args.anomaly_ratio
# model = model.to(device) 
# optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
# train(args.dataset, model, train_loader, optimizer, device, modelpath = args.modelfolder, mode= args.mode)

