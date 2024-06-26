import torch
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer
import pickle
import os


class SaveSentenceEmbeddings:

    def __init__(self,all_sentences, batch_size):
        self.sbert_model = SentenceTransformer('stsb-roberta-base', device = torch.cuda.current_device())
        self.all_sentences = all_sentences
        self.batch_size = batch_size

    def add_batch_encodings(self,d, batch):
        embs = self.sbert_model.encode(batch)
        for i in range(len(batch)):
            d[batch[i]] =  embs[i]


    def save_embeddings_dict(self,path):
        iterations = len(self.all_sentences) // self.batch_size
        cur_idx = 0
        d = {}
        for i in range(iterations):
            print(f"started batch {i}")
            batch = self.all_sentences[cur_idx : cur_idx + self.batch_size]
            cur_idx += self.batch_size
            self.add_batch_encodings(d, batch)
        #adding the last batch
        self.add_batch_encodings(d, self.all_sentences[cur_idx:])
        with open(path,"wb") as f:
            pickle.dump(d,f)


class SaveOrLoadEmbeddings:

        def __init__(self, all_sentences, batch_size, save_dir):
            self.all_sentences = all_sentences
            self.batch_size = batch_size
            self.save_dir = save_dir

        def save_or_load_embeddings(self, size):
            path = self.create_path(size)
            if os.path.exists(path):
                d =  self.load_embeddings(path)
            else:
                sentences = self.all_sentences[:size]
                SSE = SaveSentenceEmbeddings(sentences, self.batch_size)
                SSE.save_embeddings_dict(path)
                d =  self.load_embeddings(path)
            index_to_key, value_list = self.convert_dict_with_index(d)
            return d, index_to_key, value_list

        def convert_dict_with_index(self,d):
            index_to_key = {}
            value_list = []
            for i, x in enumerate(d):
                index_to_key[i] = x
                value_list.append(d[x])
            return index_to_key, value_list

        def create_path(self, size):
            path = f"{self.save_dir}\\sentence_embeddings_{size}.pkl"
            return path

        def load_embeddings(self, path):
            with open(path, "rb") as f:
                return pickle.load(f)