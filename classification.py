from __future__ import print_function

import warnings

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from metrics import clustering_metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
import scipy.sparse as sp

warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak on Windows with MKL")

def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(solver='liblinear', multi_class='ovr')
    log.fit(train_embeds, train_labels)
    predict = (log.predict(test_embeds)).tolist()
    accuracy = accuracy_score(test_labels, predict)
    print("Test Accuracy:", accuracy)
    return accuracy

def classify(embeds, dataset, per_class):

    label_file = open("data/{}{}".format(dataset,"_labels.txt"), 'r')
    label_text = label_file.readlines()
    labels = []
    for line in label_text:
        if line.strip('\n'):
            line = line.strip('\n').split(' ')
            labels.append(int(line[1]))
    label_file.close()
    labels = np.array(labels)
    train_file = open("data/{}/{}/train_text.txt".format(dataset, per_class), 'r')
    train_text = train_file.readlines()
    train_file.close()
    test_file = open( "data/{}/{}/test_text.txt".format(dataset, per_class), 'r')
    test_text = test_file.readlines()
    test_file.close()
    ave = []
    for k in range(50):
        train_ids = eval(train_text[k])
        test_ids = eval(test_text[k])
        train_labels = [labels[i] for i in train_ids]
        test_labels = [labels[i] for i in test_ids]
        train_embeds = embeds[[id for id in train_ids]]
        test_embeds = embeds[[id for id in test_ids]]
        acc = run_regression(train_embeds, train_labels, test_embeds, test_labels)
        ave.append(acc)
    print(np.mean(ave)*100)
    print(np.std(ave)*100)


def clustering(embeds, dataset):
    label_file = open("data/{}{}".format(dataset, "_labels.txt"), 'r')
    label_text = label_file.readlines()
    labels = []
    for line in label_text:
        if line.strip('\n'):
            line = line.strip('\n').split(' ')
            labels.append(int(line[1]))
    label_file.close()
    labels = np.array(labels)
    rep = 10
    u = embeds
    k = len(np.unique(labels))
    ac = np.zeros(rep)
    nm = np.zeros(rep)
    f1 = np.zeros(rep)

    intra_distances = []
    inter_distances = []

    for i in range(rep):
        kmeans = KMeans(n_clusters=k).fit(u)
        predict_labels = kmeans.predict(u)

        # 计算簇内距离（intra-cluster distance）
        intra_distances.append(kmeans.inertia_ / len(u))

        # 计算簇间距离（inter-cluster distance）
        centroids = kmeans.cluster_centers_
        inter_cluster_distances = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i != j:
                    distance = np.linalg.norm(centroids[i] - centroids[j])
                    inter_cluster_distances[i, j] = distance
        inter_distances.append(np.mean(inter_cluster_distances))

        cm = clustering_metrics(labels, predict_labels)
        ac[i], nm[i], f1[i] = cm.evaluationClusterModelFromLabel()

    print("Accuracy mean:", np.mean(ac))
    print("Accuracy std:", np.std(ac))
    print("NMI mean:", np.mean(nm))
    print("NMI std:", np.std(nm))
    print("F1 mean:", np.mean(f1))
    print("F1 std:", np.std(f1))

    print("Intra-cluster distance mean:", np.mean(intra_distances))
    print("Intra-cluster distance std:", np.std(intra_distances))
    print("Inter-cluster distance mean:", np.mean(inter_distances))
    print("Inter-cluster distance std:", np.std(inter_distances))
