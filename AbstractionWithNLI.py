#imports
from sentence_transformers import CrossEncoder
import torch
import numpy as np


class AbstractionBetweenGraphs:

    def __init__(self, graph1_nodes, graph2_nodes, threshold):
        self.graph1_nodes = graph1_nodes
        self.graph2_nodes = graph2_nodes
        self.nli_model = self.get_nli_model()
        self.label_mapping = ['contradiction', 'entailment', 'neutral']
        self.graph1_sentences = self.form_sentences(self.graph1_nodes)
        self.graph2_sentences = self.form_sentences(self.graph2_nodes)
        self.threshold = threshold
        self.softmax = torch.nn.Softmax()

    def check_entailment_relations(self):
        node_to_neighbor = {}
        for node in self.graph1_nodes:
            node_to_neighbor[node] = []
        for node in self.graph2_nodes:
            node_to_neighbor[node] = []
        for node in self.graph1_nodes:
            sentences_to_check = [[self.graph1_sentences[node], self.graph2_sentences[x]] for x in self.graph2_sentences]
            scores = self.nli_model.predict(sentences_to_check)
            if self.threshold:
                scores = np.array(self.softmax(torch.tensor(scores)))
                max_scores = scores.max(axis=1)
                labels = [self.label_mapping[score_max] if max_scores[i] > self.threshold else "netural" for
                          i, score_max in enumerate(scores.argmax(axis=1))]
            else:
                labels = [self.label_mapping[score_max] for score_max in scores.argmax(axis=1)]
            for i,label in enumerate(labels):
                if label == "entailment":
                    node_to_neighbor[node].append(self.graph2_nodes[i])
        for node in self.graph2_nodes:
            sentences_to_check = [[self.graph2_sentences[node], self.graph1_sentences[x]] for x in self.graph1_sentences]
            scores = self.nli_model.predict(sentences_to_check)
            if self.threshold:
                scores = np.array(self.softmax(torch.tensor(scores)))
                max_scores = scores.max(axis=1)
                labels = [self.label_mapping[score_max] if max_scores[i] > self.threshold else "netural" for
                          i, score_max in enumerate(scores.argmax(axis=1))]
            else:
                labels = [self.label_mapping[score_max] for score_max in scores.argmax(axis=1)]
            for i,label in enumerate(labels):
                if label == "entailment":
                    node_to_neighbor[node].append(self.graph1_nodes[i])
        return node_to_neighbor




    def get_nli_model(self):
        return CrossEncoder('cross-encoder/nli-deberta-base', device=torch.cuda.current_device())

    def form_sentences(self, nodes):
        sentences = {}
        for node in nodes:
            sentences[node] = self.form_sentence(node.get_random_point())
        return sentences


    def form_sentence(self, sentence):
        return f"The device is able {sentence}"

class AbstractionWithClusters:

    def __init__(self, clusters, prefix, threshold  = None):
        self.clusters = clusters
        self.get_nli_model()
        self.prefix = prefix
        self.label_mapping = ['contradiction', 'entailment', 'neutral']
        self.form_all_sentences()
        self.threshold = threshold
        self.softmax = torch.nn.Softmax()


    def update_clusters(self, new_clusters):
        self.clusters = new_clusters
        self.form_all_sentences()


    def get_nli_model(self):
        if type == "nli_deberta":
            self.nli_model =  CrossEncoder('cross-encoder/nli-deberta-base', device=torch.cuda.current_device())



    def form_sentence(self, sentence):
        return f"{self.prefix} {sentence}"

    def form_all_sentences(self):
        self.formed_sentences = [self.form_sentence(cluster.get_random_point()) for cluster in self.clusters ]


    def check_entailment_for_cluster(self, cluster, low = 0):
        sentence1 = self.form_sentence(cluster.get_random_point())
        batch_formed_sentences = self.formed_sentences[low:]
        sentences_to_check = [(sentence1, sentence2) for sentence2 in batch_formed_sentences]
        scores = self.nli_model.predict(sentences_to_check)
        scores = np.array(self.softmax(torch.tensor(scores)))
        if self.threshold:

            max_scores = scores.max(axis = 1)
            labels = [(self.label_mapping[score_max], max_scores[i]) if max_scores[i] > self.threshold else ("netural", 0) for i,score_max in enumerate(scores.argmax(axis = 1))]
        else:
            labels = [(self.label_mapping[score_max], score_max) for score_max in scores.argmax(axis = 1)]
        return labels

    def find_abstractions_with_nli(self):
        cluster_to_more_abstract = {}
        cluster_to_less_abstract = {}
        edge_to_weight = {}
        for j,cluster in enumerate(self.clusters):
            print(f"starting with the {j}-th iteration of entailment checking")
            entailment_labels = self.check_entailment_for_cluster(cluster)
            for i,label in enumerate(entailment_labels):
                if label[0] == "entailment" and i != j:
                    current_cluster = self.clusters[i]
                    if current_cluster not in cluster_to_less_abstract:
                        cluster_to_less_abstract[current_cluster] = []
                    cluster_to_less_abstract[current_cluster].append(cluster)
                    edge_to_weight[(cluster, current_cluster)] = label[1]
                    if cluster not in cluster_to_more_abstract:
                        cluster_to_more_abstract[cluster] = []
                    cluster_to_more_abstract[cluster].append(current_cluster)
        cluster_to_more_abstract = self.add_no_neighbor_clusters(cluster_to_more_abstract)
        cluster_to_less_abstract = self.add_no_neighbor_clusters(cluster_to_less_abstract)
        return cluster_to_more_abstract, cluster_to_less_abstract, edge_to_weight

    def add_no_neighbor_clusters(self, d):
        for cluster in self.clusters:
            if cluster not in d:
                d[cluster] = []
        return d

    def check_clusters(self, cluster1, cluster2):
        sentence1 = self.form_sentence(cluster1.get_random_point())
        sentence2 = self.form_sentence(cluster2.get_random_point())

class AbstractionWithClustersMNLI(AbstractionWithClusters):


    def __init__(self, clusters, prefix, threshold  = None):
        super().__init__(clusters, prefix,threshold)
        self.label_mapping = ["entailment", "neutral", "contradiction"]

    def get_nli_model(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        #self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(torch.cuda.current_device())



    def check_entailment_for_cluster(self, cluster, low = 0, high = None):
        sentence1 = self.form_sentence(cluster.get_random_point())
        if high:
            batch_formed_sentences = self.formed_sentences[low:high]
        else:
            batch_formed_sentences = self.formed_sentences[low:]
        first_sentences = [sentence1 for _ in range(len(batch_formed_sentences))]
        labels = []
        bs = 200
        steps = len(batch_formed_sentences) // bs
        remainder = len(batch_formed_sentences) % bs
        for it in range(steps):
            inputs = self.tokenizer(first_sentences[it*bs:(it+1)*bs], batch_formed_sentences[it*bs:(it+1)*bs],
                                    truncation = True, return_tensors  = "pt", padding = True, max_length = 100)
            # inputs = self.tokenizer(first_sentences, batch_formed_sentences, truncation = True, return_tensors  = "pt", padding = True)\
            #     .to(torch.cuda.current_device())
            output = self.nli_model(inputs["input_ids"])
            scores = torch.softmax(output["logits"], -1)
            max_scores, argmax_scores = scores.max(axis=1)
            if self.threshold:

                labels.extend([(self.label_mapping[score_max], max_scores[i].item()) if max_scores[i] > self.threshold else ("netural", 0)
                               for i,score_max in enumerate(argmax_scores)])
            else:
                labels.extend([(self.label_mapping[score_max], score_max) for score_max in argmax_scores])
        inputs = self.tokenizer(first_sentences[steps * bs:], batch_formed_sentences[steps * bs:],
                                truncation=True, return_tensors="pt", padding=True, max_length = 100)
        # inputs = self.tokenizer(first_sentences, batch_formed_sentences, truncation = True, return_tensors  = "pt", padding = True).to(torch.cuda.current_device())
        output = self.nli_model(inputs["input_ids"])

        scores = torch.softmax(output["logits"], -1)
        max_scores, argmax_scores = scores.max(axis=1)
        if self.threshold:
            labels.extend(
                [(self.label_mapping[score_max], max_scores[i].item()) if max_scores[i] > self.threshold else ("netural", 0)
                 for i, score_max in enumerate(argmax_scores)])
        else:
            labels.extend([(self.label_mapping[score_max], score_max) for score_max in argmax_scores])
        return labels