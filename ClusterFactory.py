import numpy as np
class Cluster:

    def __init__(self, points, point_embeddings, id):
        self.points = points
        self.point_embeddings = point_embeddings
        self.id = id

    def get_cluster_points(self):
        return self.points

    def get_id(self):
        return self.id

    def get_point_embeddings(self):
        return self.point_embeddings

    def get_random_point(self):
        #rand_int  = np.random.randint(0,len(self.points))
        return self.points[0]



class ClusterFactory:

    @staticmethod
    def create_clusters(index_to_label, index_to_sentence, sentence_to_embedding):
        #creating a mapping between labels and sentences
        label_to_sentences = {}
        label_to_embeddings = {}
        for index in index_to_sentence:
            label  = index_to_label[index]
            if label != -1:
                if label not in label_to_sentences:
                    label_to_sentences[label] = []
                    label_to_embeddings[label] = []
                sentence = index_to_sentence[index]
                label_to_sentences[label].append(sentence)
                label_to_embeddings[label].append(sentence_to_embedding[sentence])
        clusters = [Cluster(label_to_sentences[label], label_to_embeddings[label], i) for i,label in enumerate(label_to_sentences)]
        return clusters
