import markdownify

from Serializer import Serializer
import ClusterVisualizer

def draw_ground_truth(edges_list, nodes_list, fig_path):
    node_to_neighbors = {}
    edge_to_weight = {}
    node_to_level = {node : 1 for node in nodes_list}
    for node in nodes_list:
        node_to_neighbors[node] = []
    id_to_nodes = {(node.get_id() + 1) : node for node in nodes_list}
    for edge in edges_list:
        node1 = id_to_nodes[edge[0]]
        node2 = id_to_nodes[edge[1]]
        node_to_neighbors[node1].append(node2)
        edge_to_weight[(node1, node2)] = 1

    ClusterVisualizer.ClusterVisualizer.draw_colored_cluster_graph_with_pyvis(nodes_list, node_to_neighbors, edge_to_weight,node_to_level, fig_path, title = None)



def calibrate_neighbor_dict(d, index_to_cluster):
    new_dict = {}
    for x in d:
        new_dict[index_to_cluster[x.get_id()]] = [index_to_cluster[y.get_id()] for y in d[x]]
    return new_dict

def get_graph_edges(neighbor_dict):
    graph_edges = []
    id_to_node  = {}
    for node in neighbor_dict:
        node_id = node.get_id() + 1
        id_to_node[node_id] = node
        for neighbor in neighbor_dict[node]:
            neighbor_id = neighbor.get_id() + 1
            graph_edges.append((node_id, neighbor_id))
    return graph_edges, id_to_node

def read_edges_file(fname):
    edges = []
    with open(fname) as f:
        for line in f.readlines():
            lst = line[:-1].split(",")
            edges.append((int(lst[0]),int(lst[1])))
    return edges





def calculate_write_edges_metrics(fname,id_to_node,predicted_edges, graph_edges):
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0
    with open(fname, "w") as f:
        f.write("Predicted edges which aren't true:\n")
        i = 0
        for edge in predicted_edges:
            if edge in graph_edges:
                true_positives += 1
            else:
                f.write(f"{i+1}. {id_to_node[edge[0]].get_random_point()}----> {id_to_node[edge[1]].get_random_point()}\n")
                i += 1
                false_positives += 1
        f.write("\n\n")
        f.write("GT edges which weren't found:\n")
        i = 0
        for edge in graph_edges:
            if edge not in predicted_edges:
                false_negatives += 1
                f.write(f"{i + 1}. {id_to_node[edge[0]].get_random_point()}----> {id_to_node[edge[1]].get_random_point()}\n")
                i += 1
        recall = true_positives / len(graph_edges)
        precision = true_positives / (true_positives + false_positives)
        f.write("\n\n\n")
        f.write(f"The number of GT edges is {len(edges_list)}\n")
        f.write(f"The number of predicted edges is {len(predicted_edges)}\n")
        f.write(f"recall is {recall}\n")
        f.write(f"precision is {precision}\n")
    return recall, precision

def plot_scatter_for_stat(figname, title, prefix_to_info, prefix_to_color):
    import matplotlib.pyplot as plt
    plt.figure()
    for prefix in prefix_to_info:
        color = prefix_to_color[prefix]
        info = prefix_to_info[prefix]
        x = [i[0] for i in info]
        recall = [i[1] for i in info]
        precision = [i[2] for i in info]
        plt.scatter(x, recall, color = color, marker="o", label=f"recall, prefix = {prefix}")
        plt.scatter(x, precision, color = color, marker="x", label = f"precision, prefix = {prefix}")
        plt.xlabel("entailement thershold")
        plt.ylabel("recall \ precision")
        plt.title(title)
        plt.legend()
    plt.savefig(figname)
    plt.close()




def get_summary_fname(entail_prefix, entailment_threshold,stat):

    return f"cluster_visualization\\edge_evaluation\\{entail_prefix}_{entailment_threshold}_{stat}.txt"


if __name__ == "__main__":
    entail_threshs =  [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    prefixes = ["I want"]
    status = ["full", "nodouble","nocycle"]
    prefix_to_color = {"I want" : "red", "I have": "blue", "The patent provides" : "green"}
    stat_to_prefix_to_info = {}
    for stat in status:
        stat_to_prefix_to_info[stat] = {}
        for entail_prefix in prefixes:
            stat_to_prefix_to_info[stat][entail_prefix] = []
            for entailment_threshold in entail_threshs:
                num_points = 27
                num_loose_clusters = 1
                min_height = 0
                max_height = 0
                max_dist_from_highest = 0
                max_out_degree = 0
                num_paths = 0
                inter_entailment_threshold = 0
                save_gt = True
                # loading the dictionaries for the parameters given
                reverse_top_order_path = f"Clusters\\reverse_top_orders\\label_to_top_order_{entail_prefix}_0.2_{num_points}_{entailment_threshold}_v3"
                label_to_top_order = Serializer.load_dict(reverse_top_order_path)
                if stat == "full":
                    neighbor_dict_path = f"Clusters\\neighbor_dict\\label_to_full_neighbor_dict_{entail_prefix}_0.2_{num_points}_{entailment_threshold}_v3"
                    label_to_neighbor_dict = Serializer.load_dict(neighbor_dict_path)
                    edges_fname = "toy_patents_edges.txt"
                elif stat == "nocycle":
                    neighbor_dict_path = f"Clusters\\neighbor_dict\\label_to_neighbor_dict_{entail_prefix}_0.2_{num_points}_{entailment_threshold}_v3"
                    label_to_neighbor_dict = Serializer.load_dict(neighbor_dict_path)
                    edges_fname = "toy_patents_edges_trimmed.txt"
                elif stat == "nodouble":
                    neighbor_dict_path = f"Clusters\\neighbor_dict\\label_to_nodouble_neighbor_dict_{entail_prefix}_0.2_{num_points}_{entailment_threshold}_v3"
                    label_to_neighbor_dict = Serializer.load_dict(neighbor_dict_path)
                    edges_fname = "toy_patents_edges.txt"
                cluster_save_path = f"clusters\\abs_clusters\\label_to_index_to_cluster_{entail_prefix}_0.2_{num_points}_{entailment_threshold}_v3"
                label_to_index_to_cluster = Serializer.load_dict(cluster_save_path)
                label_to_interconnecting_nodes = {}
                label_to_level_to_interconnecting_nodes = {}
                index_to_cluster = label_to_index_to_cluster[0]
                # getting the neighbor dict, calibrating it to contain the same clusters
                neighbor_dict = label_to_neighbor_dict[0]
                calibrated_neighbor_dict = calibrate_neighbor_dict(neighbor_dict, index_to_cluster)

                #reading the edges dict


                edges_list = read_edges_file(edges_fname)
                graph_edges, id_to_node = get_graph_edges(calibrated_neighbor_dict)
                #getting file name
                summary_fname = get_summary_fname(entail_prefix, entailment_threshold, stat)
                recall, precision = calculate_write_edges_metrics(summary_fname, id_to_node,graph_edges, edges_list)
                stat_to_prefix_to_info[stat][entail_prefix].append((entailment_threshold, recall, precision))
                if save_gt:
                    if stat == "nocycle":
                        figname = "toy_GT_trimmed.html"
                    else:
                        figname = "toy_GT_full.html"
                    nodes_list = list(calibrated_neighbor_dict.keys())
                    draw_ground_truth(edges_list, nodes_list, figname)
                print(f"recall is {recall}")
                print(f"precision is  {precision}")
                print(f"total number of ground truth edges is {len(edges_list)}")
                print(f"total number of predicted edges is {len(graph_edges)}")
                #setup_to_metrics[(entailment_threshold, entail_prefix, stat)] = (recall, precision)
    # with open("cluster_visualization\\edge_evaluation\\all_metrics.txt", "w") as f:
    #     for setup in setup_to_metrics:
    #         recall, precision = setup_to_metrics[setup]
    #         f.write(f"Threshold {setup[0]}, prefix:{ setup[1]}, stat: {setup[2]} : recall :{recall}, precision: {precision}\n\n")
    for stat in stat_to_prefix_to_info:
        title = f"recall/precision for the status {stat}"
        d = stat_to_prefix_to_info[stat]
        recall_figname = f"cluster_visualization\\edge_evaluation\\only1_recall_precision_{stat}.png"
        plot_scatter_for_stat(recall_figname, title, d, prefix_to_color)