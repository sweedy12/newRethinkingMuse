#imports
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import pickle
from  PurposeReader import PurposeReader
import torch
from sentence_transformers import CrossEncoder
import argparse
import SaveSentenceEmbeddings
from ClusterFactory import ClusterFactory
import ClusterVisualizer
import AbstractionWithNLI
from DAGUtils import CycleBreaker, LongestPathAlgorithms, DFSCycleBreaker, NetworkxGraphUtils
import Serializer
from CreatePurposeGraph import HierarchyFinder

class ClusterPoints:

    def __init__(self, cluster_func, points_to_cluster):
        self.cluster_func = cluster_func
        self.points_to_cluster = points_to_cluster
        self.create_index_to_point()

    def create_index_to_point(self):
        self.index_to_point = {i : x for i,x in enumerate(self.points_to_cluster)}

    def get_index_to_point(self):
        return self.index_to_point

    def cluster_points(self):
        """

        :return:  A mapping between the point embeddings to their cluster label
        """
        clusters = self.cluster_func.fit(self.points_to_cluster)
        index_to_label = {}
        for i,label in enumerate(clusters.labels_):
            index_to_label[i] = label

        #emb_to_label = {self.points_to_cluster[i] : clusters.labels_[i]  for i in range(len(self.points_to_cluster))}
        return index_to_label,clusters #

def get_clustering_function(args):
    if args.clustering_type == "dbscan":
        return DBSCAN(eps=args.eps, metric="cosine", min_samples=args.min_samples, n_jobs=-1)
    elif args.clustering_type == "kmeans":
        return KMeans(n_clusters=args.num_clusters)
    elif args.clustering_type == "agglomerative":
        if args.distance_threshold:
            print("hey")
            return AgglomerativeClustering(n_clusters=None,
                                           distance_threshold=args.distance_threshold, compute_full_tree = True,
                                           metric = "cosine", linkage="complete")

        else:
            return AgglomerativeClustering(n_clusters=args.num_clusters,
                                       distance_threshold=args.distance_threshold, metric = "cosine")
def get_loose_clustering_function(args):
    if args.loose_clustering_type == "dbscan":
        return DBSCAN(eps=args.eps, metric="cosine", min_samples=args.min_samples, n_jobs=-1)
    elif args.loose_clustering_type == "kmeans":
        return KMeans(n_clusters=args.loose_num_clusters)
    elif args.loose_clustering_type == "agglomerative":
        if args.distance_threshold:
            return AgglomerativeClustering(n_clusters=None,
                                           distance_threshold=args.distance_threshold, compute_full_tree = True,
                                           metric = "cosine", linkage="complete")

        else:
            return AgglomerativeClustering(n_clusters=args.num_clusters,
                                       distance_threshold=args.distance_threshold, metric = "cosine")

def get_clustering_kwargs(args):
    if not args.entailment_threshold:
        threshold = 0
    else:
        threshold  = args.entailment_threshold

    if args.clustering_type == "dbscan":
        return {"eps" : args.eps, "min_samples" : args.min_samples, "entailment_thresh" : threshold, "nli" : args.nli_model}
    elif args.clustering_type == "kmeans":
        return {"n_clusters" : args.num_clusters, "entailment_thresh" : threshold, "nli" : args.nli_model}
    elif args.clustering_type == "agglomerative":
        if args.distance_threshold:
            return {"n_clusters": 0, "distance_threshold": args.distance_threshold, "entailment_thresh" : threshold, "nli" : args.nli_model}
        return {"n_clusters" : args.num_clusters, "distance_threshold" : 1, "entailment_thresh" : threshold, "nli" : args.nli_model}


def get_clustering_figure_path(dir, type, size, longest,prefix,loose_label, **kwargs):
    longest_str = ""

    if longest:
        longest_str = "only_longest_"
    final_path = f"{dir}\\graph_visulization_{longest_str}{type}_size_{size}_prefix_{prefix}"
    for arg in kwargs:
        cur_arg = kwargs[arg]
        if "." in str(cur_arg):
            cur_arg = str(cur_arg).replace(".", "")
        final_path += f"{arg}_{cur_arg}"
    final_path += f"_lcluster_{loose_label}"
    final_path += ".html"
    return final_path

def get_clustering_information_path(dir, type, size, prefix,loose_label, **kwargs):
    final_path = f"{dir}\\{type}_size_{size}_prefix_{prefix}_"
    for arg in kwargs:
        cur_arg = kwargs[arg]
        if "." in str(cur_arg):
            cur_arg = str(cur_arg).replace(".","")
        final_path += f"{arg}_{cur_arg}"
    final_path += f"_lcluster_{loose_label}"
    return final_path

def get_abs_clustering_information_path(dir, type, size, longest, prefix, loose_label, **kwargs):
    longest_str = ""
    if longest:
        longest_str = "non_longest_"
    final_path = f"{dir}\\abs_clustering_{longest_str}{type}_size_{size}__prefix_{prefix}_"
    for arg in kwargs:
        cur_arg = kwargs[arg]
        if "." in str(cur_arg):
            cur_arg = str(cur_arg).replace(".", "")
        final_path += f"{arg}_{cur_arg}"
    final_path += f"_lcluster_{loose_label}"
    return final_path

def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", type=str, default="embeddings")
    parser.add_argument("--entail_prefix", type=str, default="The device is able to")
    parser.add_argument("--clustering_type", type=str, default="dbscan")
    parser.add_argument("--loose_clustering_type", type=str, default="dbscan")
    parser.add_argument("--cluster_vis_dir", type=str, default="cluster_visualization")
    parser.add_argument("--nli_model", type=str, default="deberta_v2")
    parser.add_argument("--patents_dir", type=str,
                        default="1_milion_gpt3_tagged_patents-20240207T140452Z-001\\1_milion_gpt3_tagged_patents\\")
    parser.add_argument("--num_of_points", type=int, default=1000)
    parser.add_argument("--embedding_batch_size", type=int, default=30)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--distance_threshold", type=float, default=None)
    parser.add_argument("--entailment_threshold", type=float, default=None)
    parser.add_argument("--min_samples", type=int, default=1)
    parser.add_argument("--num_clusters", type=int, default=200)
    parser.add_argument("--loose_num_clusters", type=int, default=200)
    parser.add_argument("--visualize_clusters", action="store_true")
    parser.add_argument("--visualize_abs_clusters", action="store_true")
    parser.add_argument("--remove_doubles_before", action="store_true")
    parser.add_argument("--clustering_only", action="store_true")
    return parser


def remove_edges_from_dict(node_to_neighbors, edges_to_remove):
    new_node_to_neighbors = {}
    for node in node_to_neighbors:
        new_node_to_neighbors[node] = set()
        for neighbor in node_to_neighbors[node]:
            if (node, neighbor) not in edges_to_remove:
                new_node_to_neighbors[node].add(neighbor)
    return new_node_to_neighbors

def visualize_and_write_abs_clusters(args, clusters_list, cluster_to_more_abstract, edge_to_weight, longest_cluster_to_more_abs, node_to_level, label):
    fig_path = get_clustering_figure_path(f"{args.cluster_vis_dir}\\aggressive_clusters\\full_abs_clusters\\", clustering_type,
                                          num_of_points, False,args.entail_prefix,label, **get_clustering_kwargs(args))

    longest_fig_path = get_clustering_figure_path(f"{args.cluster_vis_dir}\\aggressive_clusters\\only_longest\\", clustering_type,
                                                  num_of_points, True, args.entail_prefix,label, **get_clustering_kwargs(args))


    abs_path = get_abs_clustering_information_path(f"{args.cluster_vis_dir}\\aggressive_clusters\\full_abs_clusters\\",
                                                   clustering_type,
                                                   num_of_points, False,args.entail_prefix,label, **get_clustering_kwargs(args))
    longest_abs_path = get_abs_clustering_information_path(f"{args.cluster_vis_dir}\\aggressive_clusters\\only_longest\\"
                                                           , clustering_type,
                                                           num_of_points,True,args.entail_prefix,label, **get_clustering_kwargs(args))

    # writing clusters before & after cutting cycles and keeping only the longest paths
    ClusterVisualizer.ClusterVisualizer.write_clusters_with_abstraction(clusters_list, cluster_to_more_abstract,
                                                                        abs_path)
    ClusterVisualizer.ClusterVisualizer.write_clusters_with_abstraction(clusters_list, longest_cluster_to_more_abs,
                                                                        longest_abs_path)

    # saving the graph html file before & after cutting cycles and keeping only the longest paths
    # ClusterVisualizer.ClusterVisualizer.draw_cluster_graph_with_plotly(clusters_list, cluster_to_more_abstract,
    #                                                                    fig_path,
    #                                                                    title="agglomerative_10000_0.15")
    # ClusterVisualizer.ClusterVisualizer.draw_cluster_graph_with_plotly(clusters_list, longest_cluster_to_more_abs,
    #                                                                    longest_fig_path,
    #                                                                    title="agglomerative_10000_0.15")
    #                                                                    title="agglomerative_10000_0.15")
    ClusterVisualizer.ClusterVisualizer.draw_colored_cluster_graph_with_pyvis(clusters_list,
                                                                              cluster_to_more_abstract,
                                                                              edge_to_weight,
                                                                              node_to_level,
                                                                              fig_path,
                                                                              title=f"cluster {label} full graph")
    ClusterVisualizer.ClusterVisualizer.draw_colored_cluster_graph_with_pyvis(clusters_list, longest_cluster_to_more_abs,
                                                                              edge_to_weight,node_to_level,

                                                                              longest_fig_path, title=f"cluster {label} graph")

def prepare_loose_clusters(index_to_label, index_to_sentence, sentence_to_embedding):
    label_to_cluster_sentences = {}
    label_to_cluster_embeddings = {}
    labels = []
    for ind in index_to_label:
        label  = index_to_label[ind]
        if label not in label_to_cluster_sentences:
            label_to_cluster_sentences[label] = []
            label_to_cluster_embeddings[label] = []
            labels.append(label)
        cur_sentence = index_to_sentence[ind]
        label_to_cluster_sentences[label].append(cur_sentence)
        label_to_cluster_embeddings[label].append(sentence_to_embedding[cur_sentence])
    return labels, label_to_cluster_sentences, label_to_cluster_embeddings

def remove_double_edges(nodes_list,edge_to_weight):
    new_node_to_neighbors = {node:[] for node in nodes_list}

    for node1,node2 in edge_to_weight:
        if (node2,node1) in edge_to_weight:
            reg_weight = edge_to_weight[(node1,node2)]
            rev_weight = edge_to_weight[(node2, node1)]
            if reg_weight > rev_weight:
                new_node_to_neighbors[node1].append(node2)
        else:
            new_node_to_neighbors[node1].append(node2)
    return new_node_to_neighbors






'''
The main function does the following:
    1. performs clustering (default: k-means) to create n loose clusters
    2. For each loose cluster:
        a. perform clustering (default: agglomerative clustering) to create aggressive clusters. 
        b.find abstraction relations between the clusters using NLI
        c. Prune the graph to eliminate circles, and keep only longest paths between nodes
    3. saving visualizations of both loose, aggressive clusters
    4. saving 3 dictionaries:
        a. label (loose cluster) to neighbor dict (aggressive cluster to neighbors)
        b. label (loose cluster) to reverse topological order (important for finding height)
        c. label (loose cluster) to index_to_cluster dict (maps index to cluster)
'''
if __name__ == "__main__":

    #parsing args
    parser = add_args()
    args = parser.parse_args()
    patents_dir = args.patents_dir
    embeddings_path = args.embeddings_path
    num_of_points = args.num_of_points
    batch_size = args.embedding_batch_size
    clustering_type = args.clustering_type


    #getting purpose sentences, and their embeddings
    PR = PurposeReader()
    purpose_dict = PR.create_purpose_dict(patents_dir)
    sentences = list(purpose_dict.values())
    #loading or saving embeddings
    embeddings_loader = SaveSentenceEmbeddings.SaveOrLoadEmbeddings(sentences, batch_size, embeddings_path)
    embeddings_dict, index_to_sentence, embeddings_to_cluster = embeddings_loader.save_or_load_embeddings(num_of_points)

    #creating the loose clusters:
    loose_clustering_func = get_loose_clustering_function(args)
    loose_clusterer = ClusterPoints(loose_clustering_func, embeddings_to_cluster)
    loose_index_to_label, loose_clusters =  loose_clusterer.cluster_points()
    labels, label_to_cluster_sentences, label_to_cluster_embeddings = \
        prepare_loose_clusters(loose_index_to_label, index_to_sentence, embeddings_dict)
    label_to_top_order = {}
    label_to_neighbor_dict = {}
    label_to_index_to_cluster = {}
    label_to_full_neighbor_dict = {}
    label_to_nodouble_neighbor_dict = {}
    # clustering aggressively
    for it, label in enumerate(labels):
        label_sentences = label_to_cluster_sentences[label]
        cur_embeddings_to_cluster = label_to_cluster_embeddings[label]
        # getting the clustering function and aggressive clustering all points

        print(f"starting with dist_threshold {args.distance_threshold}")
        clustering_func = get_clustering_function(args)
        clusterer = ClusterPoints(clustering_func, cur_embeddings_to_cluster)
        index_to_label, clusters = clusterer.cluster_points()
        num_of_noise = len([x for x in list(index_to_label.values()) if x == -1])
        print(f"there are {num_of_noise} noisy points")

        # creating new index_to_sentence

        cur_index_to_sentence = {i: sentence for i, sentence in enumerate(label_sentences)}
        clusters_list = ClusterFactory.create_clusters(index_to_label, cur_index_to_sentence, embeddings_dict)
        label_to_index_to_cluster[label] = {i: x for i, x in enumerate(clusters_list)}
        if args.visualize_clusters:
            vis_path = get_clustering_information_path(f"{args.cluster_vis_dir}\\loose_clusters\\", clustering_type,
                                                       num_of_points, args.entail_prefix, label,
                                                       **get_clustering_kwargs(args))
            vis_condensed_path  = vis_path + "_condensed"
            ClusterVisualizer.ClusterVisualizer.write_clusters_to_file(clusters_list, vis_path)
            ClusterVisualizer.ClusterVisualizer.write_clusters_condensed(clusters_list, vis_condensed_path)

        thersholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        if not args.clustering_only:
            for entailment_threshold in thersholds:
                args.entailment_threshold = entailment_threshold
                print(f"starting with threshold {args.entailment_threshold}")
                print()

                #performing abstraction within a cluster
                if it == 0:
                    if args.nli_model == "v3":
                        abs_with_NLI = AbstractionWithNLI.AbstractionWithClustersMNLI(clusters_list, args.entail_prefix,
                                                                                  threshold=args.entailment_threshold)

                    else:
                        abs_with_NLI = AbstractionWithNLI.AbstractionWithClusters(clusters_list,args.entail_prefix, threshold=args.entailment_threshold)
                else:
                    abs_with_NLI.update_clusters(clusters_list)
                cluster_to_id  = {x:i for i,x in enumerate(clusters_list)}
                cluster_to_more_abstract, cluster_to_less_abstract, edge_to_weight = abs_with_NLI.find_abstractions_with_nli()
                # g = NetworkxGraphUtils.create_networkx_graph(cluster_to_more_abstract, cluster_to_id)
                nodouble_cluster_to_more_abstract = remove_double_edges(list(cluster_to_more_abstract.keys()), edge_to_weight)
                cycle_breaker = DFSCycleBreaker(cluster_to_more_abstract)
                edges_to_remove, _ = cycle_breaker.dfs_remove_back_edges()
                # nir = 1
                no_cycle_cluster_to_more_abs = remove_edges_from_dict(cluster_to_more_abstract, edges_to_remove)
                #no_cycle_cluster_to_more_abs = CycleBreaker.break_cycles(cluster_to_more_abstract)

                #pruning out circles & keeping only longest
                LPA = LongestPathAlgorithms(clusters_list, no_cycle_cluster_to_more_abs)
                longest_cluster_to_more_abs, reverse_top_order = LPA.eliminate_nonlongest_edges()
                #the abstract graph for this cluster is formed../
                label_to_neighbor_dict[label] = longest_cluster_to_more_abs
                label_to_nodouble_neighbor_dict[label] = nodouble_cluster_to_more_abstract
                label_to_full_neighbor_dict[label] = cluster_to_more_abstract
                top_order = list(reversed(reverse_top_order))
                hfinder = HierarchyFinder(longest_cluster_to_more_abs, top_order)
                node_to_level, highest_level = hfinder.find_hierarchy_in_disconnected_graph()
                label_to_top_order[label] = top_order
                for cluster in clusters_list:
                    if cluster not in cluster_to_more_abstract:
                        cluster_to_more_abstract[cluster] = set()
                if args.visualize_abs_clusters:
                    visualize_and_write_abs_clusters(args, clusters_list, cluster_to_more_abstract, edge_to_weight,longest_cluster_to_more_abs, node_to_level,  label)
                #entailment_threshold  = 0 if not args.entailment_threshold else args.entailment_threshold
                reverse_top_order_path = f"Clusters\\reverse_top_orders\\label_to_top_order_{args.entail_prefix}_{args.distance_threshold}_{args.num_of_points}_{args.entailment_threshold}_{args.nli_model}"
                Serializer.Serializer.save_dict(label_to_top_order, reverse_top_order_path)
                neighbor_dict_path = f"Clusters\\neighbor_dict\\label_to_neighbor_dict_{args.entail_prefix}_{args.distance_threshold}_{args.num_of_points}_{args.entailment_threshold}_{args.nli_model}"
                Serializer.Serializer.save_dict(label_to_neighbor_dict, neighbor_dict_path)
                nodouble_neighbor_dict_path = f"Clusters\\neighbor_dict\\label_to_nodouble_neighbor_dict_{args.entail_prefix}_{args.distance_threshold}_{args.num_of_points}_{args.entailment_threshold}_{args.nli_model}"
                Serializer.Serializer.save_dict(label_to_nodouble_neighbor_dict, nodouble_neighbor_dict_path)
                full_neighbor_dict_path = f"Clusters\\neighbor_dict\\label_to_full_neighbor_dict_{args.entail_prefix}_{args.distance_threshold}_{args.num_of_points}_{args.entailment_threshold}_{args.nli_model}"
                Serializer.Serializer.save_dict(label_to_full_neighbor_dict, full_neighbor_dict_path)
                cluster_save_path = f"clusters\\abs_clusters\\label_to_index_to_cluster_{args.entail_prefix}_{args.distance_threshold}_{args.num_of_points}_{args.entailment_threshold}_{args.nli_model}"
                Serializer.Serializer.save_dict(label_to_index_to_cluster, cluster_save_path)
