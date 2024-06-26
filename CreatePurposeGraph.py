import math

from Serializer import Serializer
from ClusterFactory import Cluster
import os
import SaveSentenceEmbeddings
from PurposeReader import PurposeReader
from DAGUtils import HierarchyFinder, NodeGetter
import AbstractionWithNLI
from ClusterVisualizer import ClusterVisualizer
from PathVisualizer import PathVisualizer, PathFinder

class ConnectGraphs:

    def __init__(self, graph1_nodes, graph2_nodes, entailment_threshold):
        self.graph1_nodes = graph1_nodes
        self.graph2_nodes = graph2_nodes
        self.entailment_threshold = entailment_threshold

    def connect_graphs(self):
        if not self.graph1_nodes or not self.graph2_nodes:
            return {}
        abs_nli = AbstractionWithNLI.AbstractionBetweenGraphs(self.graph1_nodes, self.graph2_nodes, self.entailment_threshold)
        cluster_to_more_abstract= abs_nli.check_entailment_relations()
        return cluster_to_more_abstract


def read_neighbor_dicts_from_dir(dir):
    label_to_neighbpr_dict = {}
    for file in os.listdir(dir):
        cluster_str = "lcluster_"
        sind = file.find(cluster_str)
        label = int(file[sind + len(cluster_str):])
        neighbor_dict = Serializer.load_dict(f"{dir}\\{file}")
        label_to_neighbpr_dict[label] = neighbor_dict
    return label_to_neighbpr_dict

def get_embeddings_dict():
    # getting the embedding dict
    patents_dir = "1_milion_gpt3_tagged_patents-20240207T140452Z-001\\1_milion_gpt3_tagged_patents\\"
    embeddings_path = "embeddings"
    num_of_points = 20
    batch_size = 20
    PR = PurposeReader()
    purpose_dict = PR.create_purpose_dict(patents_dir)
    sentences = list(purpose_dict.values())
    embeddings_loader = SaveSentenceEmbeddings.SaveOrLoadEmbeddings(sentences, batch_size, embeddings_path)
    embeddings_dict, index_to_sentence, embeddings_to_cluster = embeddings_loader.save_or_load_embeddings(
        num_of_points)
    return embeddings_dict

def read_clusters_from_dir(dir):
    label_to_cluster = {}
    for file in os.listdir(dir):
        cluster_str = "loose_cluster_"
        sind = file.find(cluster_str)
        label = int(file[sind + len(cluster_str):])
        embeddings_dict  = get_embeddings_dict()
        cluster_list = Serializer.load_clusters(f"{dir}\\{file}", embeddings_dict)
        label_to_cluster[label] = cluster_list
    return label_to_cluster

def read_top_order_from_dir(dir):
    label_to_reverse_top_order = {}
    for file in os.listdir(dir):
        cluster_str = "lcluster_"
        sind = file.find(cluster_str)
        label = int(file[sind + len(cluster_str):])
        embeddings_dict = get_embeddings_dict()
        cluster_list = list(reversed(Serializer.load_clusters_list(f"{dir}\\{file}")))
        label_to_reverse_top_order[label] = cluster_list
    return label_to_reverse_top_order

def replace_index_with_cluster(index_list, cluster_list):
    index_to_cluster = {i:x for i,x in enumerate(cluster_list)}
    new_list = [index_to_cluster[i] for i in index_list]
    return new_list

def calibrate_neighbor_dict(d, index_to_cluster):
    new_dict = {}
    for x in d:
        new_dict[index_to_cluster[x.get_id()]] = [index_to_cluster[y.get_id()] for y in d[x]]
    return new_dict

def calibrate_cluster_list(l, index_to_cluster):
    return [index_to_cluster[x.get_id()] for x in l]

def flatten_dict_nodes(d):
    nodes = []
    for x in d:
        for y in d[x]:
            nodes.append(y)
    return nodes

def merge_dicts(dict_list):
    merged_dict = {}
    for dict in dict_list:
        for key in dict:
            if key not in merged_dict:
                merged_dict[key] = []
            merged_dict[key].extend(dict[key])
            for neighbor in dict[key]:
                if neighbor not in dict:
                    print("inner fault")
    return merged_dict


def add_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", type=str, default="embeddings")
    parser.add_argument("--clustering_type", type=str, default="dbscan")
    parser.add_argument("--entail_prefix", type=str, default="I want")
    parser.add_argument("--loose_clustering_type", type=str, default="dbscan")
    parser.add_argument("--cluster_vis_dir", type=str, default="cluster_visualization")
    parser.add_argument("--patents_dir", type=str,
                        default="1_milion_gpt3_tagged_patents-20240207T140452Z-001\\1_milion_gpt3_tagged_patents\\")
    parser.add_argument("--num_points", type=int, default=1000)
    parser.add_argument("--embedding_batch_size", type=int, default=30)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--distance_threshold", type=float, default=None)
    parser.add_argument("--entailment_threshold", type=float, default=None)
    parser.add_argument("--inter_entailment_threshold", type=float, default=None)
    parser.add_argument("--min_samples", type=int, default=1)
    parser.add_argument("--num_clusters", type=int, default=200)
    parser.add_argument("--num_loose_clusters", type=int, default=5)
    parser.add_argument("--num_paths", type=int, default=20)
    parser.add_argument("--min_height", type=int, default=0) #if negative, marks difference from max. Else, marks level.
    parser.add_argument("--max_height", type=int, default=-1) #-1 for maximum height.
    parser.add_argument("--max_dist_from_highest", type=int, default=-1) #maximum distance from the highest in path, -1 for for max
    parser.add_argument("--max_out_degree", type=int, default=-1) #maximumout degree, -1 for for max
    parser.add_argument("--visualize_abs_clusters", action="store_true")
    return parser

def process_graph_args(arg_min, arg_max, maximal_height, max_out_degree, max_dist_from_highest):
    return_max = arg_max
    return_min = arg_min
    return_max_out_degree = max_out_degree
    return_max_dist_from_highest = max_dist_from_highest
    if return_max_out_degree == -1:
        return_max_out_degree = math.inf
    if return_max_dist_from_highest == -1:
        return_max_out_degree = math.inf
    if arg_max == -1:
        return_max = maximal_height
    if arg_min < 0:
        return_min  = return_max + arg_min
    return return_max, return_min, return_max_out_degree, return_max_dist_from_highest



if __name__ == "__main__":
    parser = add_args()
    args = parser.parse_args()
    num_points = args.num_points
    num_loose_clusters = args.num_loose_clusters
    entailment_threshold = args.entailment_threshold
    min_height = args.min_height
    max_height = args.max_height
    max_dist_from_highest = args.max_dist_from_highest
    max_out_degree = args.max_out_degree
    num_paths = args.num_paths
    inter_entailment_threshold = args.inter_entailment_threshold
    entail_prefix = args.entail_prefix

    #reading presaved dictionaries:
    inter_connection_save_path = f"cluster_visualization\\aggressive_clusters\\inter_connections_nloose_{num_loose_clusters}_" \
                                 f"npoints_{args.num_points}_thresh_{args.entailment_threshold}_" \
                                 f"leaf_min_height_{min_height}_max_height_{max_height}_outdeg_{max_out_degree}_" \
                             f"maxdist_{max_dist_from_highest}"
    inter_connection_save_path_pyvis = f"cluster_visualization\\aggressive_clusters\\inter_connections_nloose_{num_loose_clusters}_" \
                                 f"npoints_{num_points}_thresh_{entailment_threshold}_" \
                                       f"leaf_min_height_{min_height}_max_height_{max_height}_outdeg_{max_out_degree}_" \
                             f"maxdist_{max_dist_from_highest}_interentail_{inter_entailment_threshold}.html"
    entire_graph_save_path = f"cluster_visualization\\aggressive_clusters\\entire_graph_nloose_{num_loose_clusters}_" \
                                 f"npoints_{num_points}_thresh_{entailment_threshold}_" \
                                       f"min_height_{min_height}_max_height_{max_height}_outdeg_{max_out_degree}_" \
                                        f"maxdist_{max_dist_from_highest}_interentail_{inter_entailment_threshold}.html"


    #loading the dictionaries for the parameters given
    reverse_top_order_path = f"Clusters\\reverse_top_orders\\label_to_top_order_{entail_prefix}_{num_points}_{entailment_threshold}"
    label_to_top_order =Serializer.load_dict(reverse_top_order_path)
    neighbor_dict_path = f"Clusters\\neighbor_dict\\label_to_neighbor_dict_{entail_prefix}_{num_points}_{entailment_threshold}"
    label_to_neighbor_dict = Serializer.load_dict(neighbor_dict_path)
    cluster_save_path = f"clusters\\abs_clusters\\label_to_index_to_cluster_{entail_prefix}_{num_points}_{entailment_threshold}"
    label_to_index_to_cluster = Serializer.load_dict(cluster_save_path)
    label_to_interconnecting_nodes  = {}
    label_to_level_to_interconnecting_nodes = {}
    highest_levels = []
    interconnecting_nodes = []

    #going through each loose cluster, picking the nodes to check interconnections for.
    for label in label_to_top_order:
        index_to_cluster = label_to_index_to_cluster[label]
        #getting the neighbor dict, calibrating it to contain the same clusters
        neighbor_dict = label_to_neighbor_dict[label]
        calibrated_neighbor_dict = calibrate_neighbor_dict(neighbor_dict, index_to_cluster)
        label_to_neighbor_dict[label] = calibrated_neighbor_dict
        #getting and calibrating the top order
        calibrated_top_order = calibrate_cluster_list(label_to_top_order[label], index_to_cluster)
        label_to_top_order[label] = calibrated_top_order

        #saving a few paths:
        paths_save_path = f"cluster_visualization\\Paths\\{num_paths}_paths_{num_points}_label_{label}_thresh_{entailment_threshold}_" \
                          f"leaf_min_height_{min_height}_max_height_{max_height}_outdeg_{max_out_degree}_" \
                             f"maxdist_{max_dist_from_highest}_interentail_{inter_entailment_threshold}"
        # now, writing a few paths in the graph:
        path_finder = PathFinder(calibrated_neighbor_dict, calibrated_top_order, None)
        paths = path_finder.find_paths_in_graph(num_paths)
        PathVisualizer.write_paths_to_file(paths_save_path, paths)

        hfinder = HierarchyFinder(calibrated_neighbor_dict, calibrated_top_order )
        node_to_level, node_to_highest_in_path = hfinder.find_hierarchy_in_disconnected_graph()
        inter_node_getter  = NodeGetter(calibrated_neighbor_dict, node_to_level, node_to_highest_in_path)
        #processing args to fit current graph sizes:
        highest_level = max(node_to_level.values())
        cur_max_height, cur_min_height, cur_max_out_degree, cur_max_dist_from_highest = process_graph_args(min_height, max_height, highest_level, max_out_degree, max_dist_from_highest)
        print(f"max height is {cur_max_height}")
        print(f"min height is {cur_min_height}")
        print(f"max out is {cur_max_out_degree}")
        print(f"max dist is {cur_max_dist_from_highest}")
        print()
        highest_levels.append(highest_level)
        #now, getting the nodes we want to interconnect between loose cluster graphs

        #selecting leafs of certain height:
        interconnecting_cands, level_to_interconnecting_nodes = \
            inter_node_getter.find_nodes(cur_min_height, cur_max_height, cur_max_dist_from_highest, cur_max_out_degree)
        interconnecting_nodes.extend(interconnecting_cands)
        #interconnectiong_cands = hfinder.get_node_by_hierarchy_range(node_to_level, highest_level-1, highest_level)
        #optional - saving the loose graph
        # title = f"Cluster graph for cluster number {label}"
        # fig_path = f"10k_graph_cluster_{label}.html"
        # # ClusterVisualizer.draw_colored_cluster_graph_with_pyvis(calibrated_top_order,
        # #                                     calibrated_neighbor_dict,node_to_level,fig_path, title="am i nir")
        label_to_level_to_interconnecting_nodes[label] = level_to_interconnecting_nodes
        label_to_interconnecting_nodes[label] = interconnecting_cands
    print(highest_levels)
    #check for connections between the different graphs, using the interconnecting  nodes:
    inter_node_connections = {}
    seen_pairs = set()
    cluster_to_labels = {}
    #finding interconnecting abstraction relations.
    for label in label_to_top_order:
        graph1_nodes = label_to_interconnecting_nodes[label]
        for node in graph1_nodes:
            if node not in cluster_to_labels:
                cluster_to_labels[node] = label
        for label2 in label_to_top_order:
            if label2 != label:
                if (label, label2) not in seen_pairs:
                    #getting the nodes list for each loose cluster graph:
                    graph2_nodes = label_to_interconnecting_nodes[label2]
                    for node in graph2_nodes:
                        if node not in cluster_to_labels:
                            cluster_to_labels[node] = label2
                    graph_connector = ConnectGraphs(graph1_nodes, graph2_nodes, inter_entailment_threshold)
                    #updating seen pairs
                    seen_pairs.add((label, label2))
                    seen_pairs.add((label2, label))
                    #Now, check for entailments, and write the new clusters.
                    cur_abstractions = graph_connector.connect_graphs()
                    for x in cur_abstractions:
                        if x not in inter_node_connections:
                            inter_node_connections[x] = []
                        for y in cur_abstractions[x]:
                            inter_node_connections[x].append(y)
    #visualizing the connections between clusters:
    ClusterVisualizer.write_interconnecting_clusters(inter_node_connections, cluster_to_labels, inter_connection_save_path)
    fig_title = f"interconnecting_{num_points}_{entailment_threshold}_leaf_min_{min_height}_max_{max_height}_interentail_{inter_entailment_threshold}"
    ClusterVisualizer.draw_colored_interconnecting_graph(inter_node_connections, cluster_to_labels, fig_title ,inter_connection_save_path_pyvis)

    #finding the nodes we wish to abstract with LLMs



    #now, piecing the entire graph together:
    list_of_neighbor_dicts = [inter_node_connections]
    node_to_label = {}
    for label in label_to_neighbor_dict:
        list_of_neighbor_dicts.append(label_to_neighbor_dict[label])
        for x in label_to_neighbor_dict[label]:
            node_to_label[x] = label
    all_nodes_edges = merge_dicts(list_of_neighbor_dicts)
    full_fig_title = f"entire_graph_{num_points}_{entailment_threshold}_leaf_min_{min_height}_max_{max_height}_outdeg_{max_out_degree}_" \
                             f"maxdist_{max_dist_from_highest}_interentail_{inter_entailment_threshold}"
    ClusterVisualizer.draw_colored_cluster_graph_with_pyvis(list(all_nodes_edges.keys()), all_nodes_edges, node_to_label, entire_graph_save_path, title = full_fig_title)





