import torch.cuda
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans
import pickle
from  PurposeReader import PurposeReader
import torch
from sentence_transformers import CrossEncoder
import numpy as np



class Cluster:

    def __init__(self, points, abstract = None, concrete = None):
        self.points = points
        self.abstract = abstract
        self.concrete = concrete

    def get_points(self):
        return self.points
    def add_abstract(self, abstract):
        self.abstract = abstract

    def add_concrete(self, concrete):
        self.concrete = concrete

    def get_abstract(self):
        return self.abstract

    def get_concrete(self):
        return self.concrete



def load_sentence_embeddings(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def find_circles(text_to_entailments):
    has_circles = []
    for text in text_to_entailments:
        visited_nodes = set()
        nodes_to_visit = set(text_to_entailments[text])
        while nodes_to_visit:
            cur_node = nodes_to_visit.pop()
            visited_nodes.add(cur_node)
            if cur_node == text:
                has_circles.append(text)
                break
            else:
                if cur_node in text_to_entailments:
                    for x in text_to_entailments[cur_node]:
                        if x not in visited_nodes:
                            nodes_to_visit.add(x)
    return has_circles

class StatsSaver:

    @staticmethod
    def save_clusters_statistics(clusters, path):
        with open(path, "w") as f:
            cluster_to_count = {}
            for label in clusters.labels_:
                if label not in cluster_to_count:
                    cluster_to_count[label]  = 0
                cluster_to_count[label] += 1
            f.write(f"Overall, there are {len(list(cluster_to_count.keys()))} clusters. \n")
            f.write(f"The total number of points that were clustered is {np.sum(list(cluster_to_count.values()))}\n")
            f.write(f"The average number of points per cluster is {np.mean(list(cluster_to_count.values()))}\n\n\n")
            for label in cluster_to_count:
                f.write(f"cluster {label} contains {cluster_to_count[label]} examples\n")

    @staticmethod
    def save_entailment_stats(original_cluster, text_to_entailers, text_to_entailments,path):
        with open(path,"w") as f:
            f.write("Stats about the entailment properties of the cluster\n")
            f.write(f"the original cluster has {len(original_cluster)} points. \n")
            #getting number of entailing elements (a->b, a entails b).
            number_of_entailing = len(text_to_entailments)
            average_number_of_entailments = np.mean([len(text_to_entailments[x]) for x in text_to_entailments])
            non_entailing_examples = len(original_cluster) - number_of_entailing
            f.write(f"the number of points in the cluster that entail another point is {number_of_entailing},"
                    f" while {non_entailing_examples} examples do not entail others.\n")
            f.write(f"the average number of examples entailed from another example is {average_number_of_entailments}\n")

            #getting number of entailed elements (E.G. a->b, b is entailed by a)
            number_of_entailed = len(text_to_entailers)
            average_number_of_entailed = np.mean([len(text_to_entailers[x]) for x in text_to_entailers])
            non_entailed_examples = len(original_cluster) - number_of_entailed
            f.write(f"the number of examples in the cluster that are entailed by another example is {number_of_entailed}, "
                    f"while {non_entailed_examples} examples are not entailed by another.\n")
            f.write(f"For an example which is entailed by another, the average number of examples entailing it is "
                    f"{average_number_of_entailed}\n")
            #getting circles information
            circles = find_circles(text_to_entailments)
            f.write(f"The cluster also has {len(circles)} cycles in it. \n")





class ClusterPoints:

    def __init__(self, cluster_func, points_to_cluster):
        self.cluster_func = cluster_func
        self.points_to_cluster = points_to_cluster

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





class ProcessEntailmentInformation:
    def __init__(self, nli_model):
        self.nli_model = nli_model

    def create_pairs_dict(self, texts):
        pairs_to_label = {}
        label_mapping = ['contradiction', 'entailment', 'neutral']
        for i,text1 in enumerate(texts):
            print(f"starting with {i}th iteration")
            all_pairs = [(text1, text2) for text2 in texts]
            scores = self.nli_model.predict(all_pairs)
            labels = [label_mapping[score] for score in scores.argmax(axis=1)]
            for j,text2 in enumerate(texts):
                pairs_to_label[(text1,text2)] = labels[j]
        return pairs_to_label

    def save_pairs_dict_to_file(self, texts, path):
        pairs_to_label = self.create_pairs_dict(texts)
        with open(path, "wb") as f:
            pickle.dump(pairs_to_label, f)

    def pairs_dict_to_entailment_dicts(self, pairs_dict):
        #finding all texts that entail a text:
        text_to_entailers = {}
        text_to_entailments = {}
        for pair in pairs_dict:
            if pairs_dict[pair] == "entailment" and pair[0] != pair[1]:
                text1, text2 = pair
                if text2 not in text_to_entailers:
                    text_to_entailers[text2] = set()
                text_to_entailers[text2].add(text1)
                if text1 not in text_to_entailments:
                    text_to_entailments[text1] = set()
                text_to_entailments[text1].add(text2)
        return text_to_entailments, text_to_entailers


class ClusterWithNLI:

    def __init__(self, nli_model):
        self.nli_model = nli_model

    def create_clusters_with_NLI(self, text_to_entailments,text_to_entailers):
        clusters = []
        node_to_clusters = {}
        nodes_to_uncover = set(text_to_entailers.keys())
        while nodes_to_uncover: #while there are still nodes to uncover
            cur_abs = set()
            cur_concrete = set()
            cur_node = nodes_to_uncover.pop()
            if cur_node not in node_to_clusters: #this node has not been associated with a cluster before
                cur_abs.add(cur_node)
                abs_nodes_to_repeat = set()
                abs_nodes_to_repeat.add(cur_node)
                visited_abs_nodes = set()
                visited_concrete_nodes = set()
                while abs_nodes_to_repeat:
                    current_abstract = abs_nodes_to_repeat.pop()
                    visited_abs_nodes.add(current_abstract)

                    for node in text_to_entailers[current_abstract]:
                        if node not in visited_concrete_nodes:
                            visited_concrete_nodes.add(node)
                            #adding the nodes entailing cur_node to cur_concrete
                            cur_concrete.add(node)
                            #finding all nodes entailed by "node"
                            for ent_node in text_to_entailments[node]:
                                if ent_node not in visited_abs_nodes:
                                    cur_abs.add(ent_node)
                                    try:
                                        nodes_to_uncover.remove(ent_node)
                                    except:
                                        nir = 1
                                    abs_nodes_to_repeat.add(ent_node)
                                    visited_abs_nodes.add(ent_node)
                #the concrete and abstract clusters are finished
                abs_cluster = Cluster(cur_abs)
                conc_cluster = Cluster(cur_concrete)
                abs_cluster.add_concrete(conc_cluster)
                conc_cluster.add_abstract(abs_cluster)
                clusters.append(conc_cluster)
                clusters.append(abs_cluster)
                for node in cur_abs:
                    node_to_clusters[node] = abs_cluster
                for node in cur_concrete:
                    node_to_clusters[node] = conc_cluster
            else: #the abstract cluster already exists
                abs_cluster = node_to_clusters[cur_node]
                for x in abs_cluster.get_points():
                    if x in text_to_entailers:
                        for point in text_to_entailers[x]:
                            cur_concrete.add(point)
                conc_cluster = Cluster(cur_concrete)
                for node in cur_concrete:
                    node_to_clusters[node] = conc_cluster
                conc_cluster.add_abstract(abs_cluster)
                clusters.append(conc_cluster)
        return node_to_clusters, clusters


    def entailment_dicts_to_clusters(self, text_to_entailments,text_to_entailers):
        clusters = []
        abstract_clusters = []
        for text in text_to_entailers:
            abstract_cluster = set()
            abstract_cluster.add(text)
            regular_cluster = text_to_entailers[text]
            for text2 in text_to_entailers[text]:
                abstract_cluster = abstract_cluster.union(text_to_entailments[text2])
            clusters.append(regular_cluster)
            abstract_clusters.append(abstract_cluster)
        with open("abstract_cluster_trial", "w") as f:
            for i in range(len(abstract_clusters)):
                f.write(f"regular cluster is \n")
                for t in clusters[i]:
                    f.write(f"{t}\n")
                f.write("\n\n")
                f.write(f"abstract cluster is\n")
                for t in abstract_clusters[i]:
                    f.write(f"{t}\n")
                f.write("----------------------------------\n\n\n")


def get_text_to_label(label_to_emb, text_to_emb):
    final_dict = {}
    for text in text_to_emb:
        for label in label_to_emb:
            for other_emb in label_to_emb[label]:
                if np.array_equal(text_to_emb[text], other_emb):
                    final_dict[text] = label
                    break
    return final_dict

def load_pickle_dict(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_cluster_sentences(text_to_label, i):
    #reducing sentences to a single cluster
    #text_to_label = get_text_to_label(label_to_embs, d)
    texts = [text for text in text_to_label if text_to_label[text] == i]
    return texts

def convert_dict_with_index(d):
    index_to_key = {}
    value_list = []
    for i,x in enumerate(d):
        index_to_key[i] = x
        value_list.append(d[x])
    return index_to_key, value_list


class ConnectedComponents:

    def __init__(self, node_to_entailemnts):
        self.node_to_entailemnts = node_to_entailemnts

    def find_connected_components(self):
        self.index = 0
        self.node_to_index = {}
        self.node_to_lowlink = {}
        self.s = []
        self.strongly_connected_components = []
        for node in self.node_to_entailemnts:
            if node not in self.node_to_index:
                self.strongconnect(node)
        return self.strongly_connected_components


    def strongconnect(self, node):
        self.node_to_index[node] = self.index
        self.node_to_lowlink[node] = self.index
        self.index += 1
        self.s.append(node)
        if node in self.node_to_entailemnts:
            for adj_node in self.node_to_entailemnts[node]:
                if adj_node not in self.node_to_index:
                    self.strongconnect(adj_node)
                    self.node_to_lowlink[node] = min(self.node_to_lowlink[node], self.node_to_lowlink[adj_node])
                elif adj_node in self.s:
                    self.node_to_lowlink[node] = min(self.node_to_lowlink[node], self.node_to_lowlink[adj_node])
        if self.node_to_lowlink[node] == self.node_to_index[node]:
            strongly_connceted = list(self.s)
            self.strongly_connected_components.append(strongly_connceted)
            self.s = []



def draw_cluster_graph(nodes, nodes_to_entailments, fig_path):
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.DiGraph(directed = True)
    for node in nodes:
        G.add_node(node)
    for node in nodes_to_entailments:
        G.add_edges_from([(node, adj_node) for adj_node in nodes_to_entailments[node]])
    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    #nx.draw_networkx_labels(G, pos)
    plt.savefig(fig_path)
    #plt.show()
    plt.close()

if __name__ == "__main__":
    import os

    eps = 0.2
    path = "embeddings/sentence_embeddings_20000.pkl"
    dir = "1_milion_gpt3_tagged_patents-20240207T140452Z-001\\1_milion_gpt3_tagged_patents\\"
    clusters_stats_path = f"stats\\20k_clusters_stats_eps_{eps}"
    PR = PurposeReader()
    purpose_dict = PR.create_purpose_dict(dir)
    sentences = list(purpose_dict.values())[:20000]
    #saving sentence embeddings to path
    # SSE = SaveSentenceEmbeddings(sentences, 200)
    # SSE.save_embeddings_dict(path)

    #clustering sentences
    if not os.path.exists(f"stats\\eps_{eps}"):
        os.mkdir(f"stats\\eps_{eps}")
    if not os.path.exists(f"figs\\eps_{eps}"):
        os.mkdir(f"figs\\eps_{eps}")
    d = load_sentence_embeddings(path)
    index_to_text, embeddings = convert_dict_with_index(d)
    print(f"the number of sentences loaded is {len(list(d.keys()))}")
    dbscan_clusterer =  DBSCAN(eps, metric="cosine", min_samples=1, n_jobs=-1)
    CLN = ClusterPoints(dbscan_clusterer, embeddings)
    index_to_label, clusters = CLN.cluster_points()
    num_clusters = len(set(index_to_label.values()))
    text_to_label = {index_to_text[i] : index_to_label[i] for i in range(len(embeddings))}
    # #saving clusters stats
    StatsSaver.save_clusters_statistics(clusters, clusters_stats_path)

    nli_model = CrossEncoder('cross-encoder/nli-deberta-base', device=torch.cuda.current_device())
    PEI = ProcessEntailmentInformation(nli_model)
    CWNLI = ClusterWithNLI(nli_model)
    # #saving entailments stats:
    for i in range(num_clusters):
        print(f"starting with the {i}-th iteration")
        stats_path = f"stats\\eps_{eps}\\entailment_stats_cluster_{i}"
        fig_path = f"figs\\eps_{eps}\\cluster_{i}"
        current_cluster = get_cluster_sentences(text_to_label,i)
        cur_pairs_path = f"20k_all_pairs_to_label_{i+1}"
        if not os.path.exists(cur_pairs_path):
            print("not found")
            PEI.save_pairs_dict_to_file(current_cluster, cur_pairs_path)
        pairs_dict = load_pickle_dict(cur_pairs_path)
        print("loaded the pairs dict")
        print("getting entailment dicts")
        text_to_entailments, text_to_entailers = PEI.pairs_dict_to_entailment_dicts(pairs_dict)
        nodes = list(text_to_entailments.keys())
        draw_cluster_graph(nodes, text_to_entailments,  fig_path)
        print(f"saving stats for cluster {i}")
        #n2c, nli_clusters = CWNLI.create_clusters_with_NLI(text_to_entailments, text_to_entailers)
        #StatsSaver.save_entailment_stats(current_cluster, text_to_entailers, text_to_entailments, stats_path )
        #print(f"number of nli clusters is {len(nli_clusters)}")

    # # #texts = [reverse_d[emb] for emb in embs_to_label if embs_to_label[emb] == 0]
    # model = CrossEncoder('cross-encoder/nli-deberta-base', device=torch.cuda.current_device())

    # # CWNLI.save_clusters_to_file(texts, pairs_path)
    # # pairs_to_label = load_pickle_dict(pairs_path)
    # # d1, d2 = CWNLI.break_pairs_to_clusters(pairs_to_label)
    # #
    # text_to_entailers = {"y1":["x1","x2"],
    #                      "y2":["x1"],
    #                      "y3":["x1","x3"],
    #                     "y4": ["x2"],
    #                      "y5":["x4"],
    #                      "x1":["t1"],
    #                      "x3":["t2"]}
    # text_to_entailments = {"x1":["y1","y2","y3"],
    #                        "x2":["y1","y4"],
    #                        "x3":["y3"]
    #                        ,"x4":["y5"],
    #                        "t1":["x1"],
    #                        "t2":["x3",]}
    # nodes = list(text_to_entailments.keys())
    # SCC = ConnectedComponents(text_to_entailments)
    # print(SCC.find_connected_components())

    # nir = 1


