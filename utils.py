import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split

import load_process
import sys
import pickle as pkl
import networkx as nx
import math

from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -1.0).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return d_mat_inv_sqrt.dot(adj)

def normalize_adj(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

# def load_adj_neg(num_nodes, sample):
#     '''
#     adj_neg = np.zeros((num_nodes, num_nodes), dtype=float)
#     l = np.random.randint(0, num_nodes, size=num_nodes * (sample + 1))
#     t = 0
#     for i in range(num_nodes):
#         s = 0
#         adj_neg[i, i] = sample
#         while s < sample:
#             if i != l[t]:
#                 adj_neg[i, l[t]] = -1
#                 s += 1
#             t += 1
#     '''
#     col = np.random.randint(0, num_nodes, size=num_nodes * sample)
#     row = np.repeat(range(num_nodes), sample)
#     data = np.ones(num_nodes * sample)
#     adj_neg = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
#     #adj_neg = (sp.eye(adj_neg.shape[0]) * sample - adj_neg).toarray()
#     #adj_neg = (sp.eye(adj_neg.shape[0]) - adj_neg/sample).toarray()
#     #adj_neg = (adj_neg / sample).toarray()
#     adj_neg = normalize_adj(adj_neg).toarray()
#
#     return adj_neg

def load_adj_neg(num_nodes, sample):
    '''
    adj_neg = np.zeros((num_nodes, num_nodes), dtype=float)
    l = np.random.randint(0, num_nodes, size=num_nodes * (sample + 1))
    t = 0
    for i in range(num_nodes):
        s = 0
        adj_neg[i, i] = sample
        while s < sample:
            if i != l[t]:
                adj_neg[i, l[t]] = -1
                s += 1
            t += 1
    '''
    col = np.random.randint(0, num_nodes, size=num_nodes * sample)
    row = np.repeat(range(num_nodes), sample)
    index = np.not_equal(col,row)
    col = col[index]
    row = row[index]
    new_col = np.concatenate((col,row),axis=0)
    new_row = np.concatenate((row,col),axis=0)
    #data = np.ones(num_nodes * sample*2)
    data = np.ones(new_col.shape[0])
    adj_neg = sp.coo_matrix((data, (new_row, new_col)), shape=(num_nodes, num_nodes))
    #adj_neg = (sp.eye(adj_neg.shape[0]) * sample - adj_neg).toarray()
    #adj_neg = (sp.eye(adj_neg.shape[0]) - adj_neg/sample).toarray()
    #adj_neg = (adj_neg / sample).toarray()
    adj_neg = normalize_adj(adj_neg)

    return adj_neg.toarray()

# def load_adj_neg(num_nodes, sample):
#     '''
#     adj_neg = np.zeros((num_nodes, num_nodes), dtype=float)
#     l = np.random.randint(0, num_nodes, size=num_nodes * (sample + 1))
#     t = 0
#     for i in range(num_nodes):
#         s = 0
#         adj_neg[i, i] = sample
#         while s < sample:
#             if i != l[t]:
#                 adj_neg[i, l[t]] = -1
#                 s += 1
#             t += 1
#     '''
#     col = np.random.randint(0, num_nodes, size=num_nodes * sample)
#     row = np.repeat(range(num_nodes), sample)
#     index = np.greater(col,row)
#     col = col[index]
#     row = row[index]
#     new_col = np.concatenate((col,row),axis=0)
#     new_row = np.concatenate((row,col),axis=0)
#     data = np.ones(new_row.shape[0])
#     adj_neg = sp.coo_matrix((data, (new_row, new_col)), shape=(num_nodes, num_nodes))
#     # adj_neg = (sp.eye(adj_neg.shape[0]) * sample - adj_neg)
#     adj_neg = normalize_adj(adj_neg)
#     #adj_neg = (sp.eye(adj_neg.shape[0]) - adj_neg/sample).toarray()
#
#     return adj_neg.toarray()



def load_dataset(dataset_str):
    if dataset_str == 'cora_full':
        data_name = dataset_str + '.npz'
        data_graph = load_process.load_npz_to_sparse_graph("data/{}".format(data_name))
        data_graph.to_undirected()
        data_graph.to_unweighted()
        A = data_graph.adj_matrix
        X = data_graph.attr_matrix
        adj_normalized = torch.from_numpy(normalize_adj(sp.eye(A.shape[0]) + A).toarray()).float()
        X = torch.from_numpy(X.todense()).float()
    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)
        if dataset_str == 'citeseer':
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj_normalized = torch.from_numpy(normalize_adj(sp.eye(adj.shape[0]) + adj).toarray()).float()
        X = torch.from_numpy(features.todense()).float()

    return X, adj_normalized

def load_dataset_adj_lap(dataset_str):
    if dataset_str == 'cora_full':
        data_name = dataset_str + '.npz'
        data_graph = load_process.load_npz_to_sparse_graph("data/{}".format(data_name))
        data_graph.to_undirected()
        data_graph.to_unweighted()
        A = data_graph.adj_matrix
        X = data_graph.attr_matrix
        adj_normalized = torch.from_numpy(normalize_adj(sp.eye(A.shape[0]) + A).toarray()).float()
        X = torch.from_numpy(X.todense()).float()
        Laplacian = torch.from_numpy(normalize_adj(A).toarray()).float()
    elif dataset_str == 'newdata':
        # 加载Excel文件
        file_path = 'E:\\Project\\DataMining\\data\\报警统计.xlsx'
        data = pd.read_excel(file_path, header=1)  # 跳过第一行中文属性名，从第二行开始加载

        # 选择报警属性（使用英文属性名）
        alarm_attributes = [
            'fatigue_alarm', 'smoking', 'make_phone', 'over_speed', 'lighting',
            'lane_departure', 'car_too_close', 'collision', 'frequent_smoking', 'frequent_make_phone',
            'dangerous_turn', 'single_drive_over', 'today_drive_over', 'occlusion_day'
        ]

        # 初始化权重系数 alpha
        n = len(alarm_attributes)
        alphas = np.ones(n) / n

        # 将所有报警次数列转换为数值类型，非数值数据转换为零
        data[alarm_attributes] = data[alarm_attributes].apply(pd.to_numeric, errors='coerce').fillna(0)

        # 使用反对数函数计算每个属性的得分
        def score_function(x, beta=0.1):
            return np.exp(-beta * x)

        scores = {attr: score_function(data[attr]) for attr in alarm_attributes}

        # 将得分字典转换为 DataFrame
        scores_df = pd.DataFrame(scores)

        # 计算每条数据的最终得分
        final_scores = scores_df.dot(alphas)

        # 准备节点特征矩阵和邻接矩阵
        # 节点特征矩阵为加权得分
        node_features = scores_df.values

        # 使用K近邻算法构建邻接矩阵
        adjacency_matrix = kneighbors_graph(node_features, n_neighbors=5, mode='connectivity', include_self=True)
        adjacency_matrix = adjacency_matrix.toarray()

        # 转换为稀疏矩阵格式
        adjacency_matrix_sparse = csr_matrix(adjacency_matrix)

        # 计算标准化的邻接矩阵和拉普拉斯矩阵
        adj_normalized = torch.from_numpy(
            normalize_adj(sp.eye(adjacency_matrix.shape[0]) + adjacency_matrix).toarray()).float()
        Laplacian = torch.from_numpy(normalize_adj(adjacency_matrix).toarray()).float()
        X = torch.from_numpy(node_features).float()

        # 生成标签
        labels = []
        for index, score in enumerate(final_scores):
            if 0.99 <= score <= 1.0:
                labels.append(f"{index} 0")
            elif 0.9 <= score < 0.99:
                labels.append(f"{index} 1")
            elif 0.8 <= score < 0.9:
                labels.append(f"{index} 2")
            else:
                labels.append(f"{index} 3")

        # 保存标签到文件
        label_file_path = 'E:\\Project\\COLES\\data\\newdata_labels.txt'
        with open(label_file_path, 'w') as f:
            for label in labels:
                f.write(label + '\n')

        # 创建训练和测试集索引文件的函数
        def create_train_test_files(labels, dataset_name, per_class):
            labels = np.array([int(label.split()[1]) for label in labels])
            unique_labels = np.unique(labels)

            train_indices = []
            test_indices = []

            for i in range(50):
                train_idx = []
                test_idx = []

                for label in unique_labels:
                    label_indices = np.where(labels == label)[0]
                    train_idx_label, test_idx_label = train_test_split(label_indices, test_size=0.2, random_state=i)
                    train_idx.extend(train_idx_label)
                    test_idx.extend(test_idx_label)

                train_indices.append(train_idx)
                test_indices.append(test_idx)

            # 创建目录
            train_dir = f"E:\\Project\\COLES\\data\\{dataset_name}\\{per_class}"
            test_dir = f"E:\\Project\\COLES\\data\\{dataset_name}\\{per_class}"
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # 保存训练集索引
            with open(f"{train_dir}\\train_text.txt", 'w') as f:
                for train_idx in train_indices:
                    f.write(str(train_idx) + '\n')

            # 保存测试集索引
            with open(f"{test_dir}\\test_text.txt", 'w') as f:
                for test_idx in test_indices:
                    f.write(str(test_idx) + '\n')

        # 创建每类20个样本和5个样本的训练和测试集索引文件
        create_train_test_files(labels, 'newdata', 20)
        create_train_test_files(labels, 'newdata', 5)
    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)
        if dataset_str == 'citeseer':
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj_normalized = torch.from_numpy(normalize_adj(sp.eye(adj.shape[0]) + adj).toarray()).float()
        Laplacian = torch.from_numpy(normalize_adj(adj).toarray()).float()
        X = torch.from_numpy(features.todense()).float()

    return X, adj_normalized, Laplacian

def load_reddit_data_lap(data_path="data/", normalization="AugNormAdj", cuda=True):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    train_lap = adj = adj + adj.T
    adj = train_lap + sp.eye(adj.shape[0])
    train_adj = adj[train_index, :][:, train_index]
    train_lap = train_lap[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
    adj = aug_normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    train_adj = aug_normalized_adjacency(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    train_lap = normalized_adjacency(train_lap)
    train_lap = sparse_mx_to_torch_sparse_tensor(train_lap).float()
    labels = torch.LongTensor(labels)
    if cuda:
        adj = adj.cuda()
        train_adj = train_adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
    return adj, train_adj, train_lap, features, labels, train_index, val_index, test_index
