# NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from utils import load_adj_neg, load_dataset_adj_lap, visualize_embeddings_3d
from ssgc import Net
import argparse
import numpy as np

# 添加新的clustering函数
from classification import clustering

# 参数设置
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='drive',
                    help='dataset')
parser.add_argument('--seed', type=int, default=123,
                    help='seed')
parser.add_argument('--nhid', type=int, default=512,
                    help='hidden size')
parser.add_argument('--output', type=int, default=5,
                    help='output size')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=100,
                    help='maximum number of epochs')
parser.add_argument('--sample', type=int, default=5,
                    help='    ')
parser.add_argument('--num_nodes', type=int, default=185,
                    help='    ')
parser.add_argument('--num_features', type=int, default=14,
                    help='    ')

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化权重系数 alpha
n = args.num_features
alphas = np.ones(n) / n

# 加载数据集
feature, adj_normalized, lap_normalized = load_dataset_adj_lap(args.dataset, alphas)
feature = feature.to(device)
adj_normalized = adj_normalized.to(device)
lap_normalized = lap_normalized.to(device)

# 特征处理
K = 8
emb = feature
for i in range(K):
    feature = torch.mm(adj_normalized, feature)
    emb += feature
emb /= K

# 负采样
neg_sample = torch.from_numpy(load_adj_neg(args.num_nodes, args.sample)).float().to(device)

# 模型定义
model = Net(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
model.train()

# 训练过程
for epoch in range(args.epochs):
    optimizer.zero_grad()
    out = model(emb)
    loss = (torch.trace(torch.mm(torch.mm(torch.transpose(out, 0, 1), neg_sample), out)) - torch.trace(
        torch.mm(torch.mm(torch.transpose(out, 0, 1), lap_normalized), out))) / out.shape[0]
    print(loss)
    loss.backward()
    optimizer.step()

# 获取嵌入
emb = model(emb).cpu().detach().numpy()
np.save('embedding.npy', emb)

# 聚类任务
result_map = clustering(emb, args.dataset)


