from Serializer import Serializer
import ClusterVisualizer
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

def read_clusters(fname):
    clusters = []
    with open(fname) as f:
        for line in f.readlines():
            cur = line.split("%%%")
            clusters.append(cur[:-1])
    return clusters

def get_sentence_to_id(clusters_list, shift = False):
    sentence_to_id = {}
    for i,cluster in enumerate(clusters_list):
        for sentence in cluster:
            if shift:
                sentence_to_id[sentence] = i+100
            else:
                sentence_to_id[sentence] = i
    return sentence_to_id

def turn_clusters_list_to_indices(clusters_list, sentence_to_id):
    index_list = []
    for cluster in clusters_list:
        index_list.append(np.array([sentence_to_id[sentence] for sentence in cluster]))
    return np.array(index_list)

def calculate_purity(predicted_index_list):
    sum_of_max = 0
    total_sum = 0
    for i in range(predicted_index_list.shape[0]):
        t = np.bincount(predicted_index_list[i])
        sum_of_max += np.max(t)
        total_sum += predicted_index_list[i].shape[0]
    return sum_of_max / total_sum


def calculate_nmi(real_sentence_to_id, predicted_sentence_to_id):
    sentence_list = list(real_sentence_to_id.keys())
    real_labels = [real_sentence_to_id[sent] for sent in sentence_list]
    predicted_labels = [predicted_sentence_to_id[sent] for sent in sentence_list]
    return normalized_mutual_info_score(real_labels, predicted_labels)

def get_predicted_name(distance_threshold):
    dist_thresh_str = str(distance_threshold).replace(".","")
    predicted_fname = f"cluster_visualization\\loose_clusters\\agglomerative_size_27_prefix_I want_n_clusters_0distance_threshold_{dist_thresh_str}entailment_thresh_09nli_v3_lcluster_0_condensed"
    return predicted_fname

def plot_nmi_purity_measures(thresholds, nmis, purities,title,figname):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(thresholds, nmis, color = "green", marker = "o", label = "NMI")
    plt.scatter(thresholds, purities, color = "red", marker = "x", label = "Purity")
    plt.xlabel("Distance threshold")
    plt.ylabel("NMI\purity")
    plt.title(title)
    plt.legend()
    plt.savefig(figname)
    plt.close()

dist_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
all_nmis = []
all_purities = []

for dist_threshold in dist_thresholds:

    predicted_fname = get_predicted_name(dist_threshold)
    real_fname = "real_clusters"

    predicted_clusters = read_clusters(predicted_fname)
    real_clusters = read_clusters(real_fname)
    predicted_sentence_to_id = get_sentence_to_id(predicted_clusters)
    real_sentence_to_id = get_sentence_to_id(real_clusters)
    predicted_index_list = turn_clusters_list_to_indices(predicted_clusters, real_sentence_to_id)
    nmi = calculate_nmi(real_sentence_to_id, predicted_sentence_to_id)
    all_nmis.append(nmi)
    print(f"nmi is {nmi}")

    purity = calculate_purity(predicted_index_list)
    all_purities.append(purity)
    print(f"purity is {purity}")
title = "NMI and Purity measures as a function of clustering distance threshold"
figname = "cluster_visualization\\loose_clusters\\purity_nmi_graph.png"
plot_nmi_purity_measures(dist_thresholds, all_nmis, all_purities,title, figname)
x = 1



