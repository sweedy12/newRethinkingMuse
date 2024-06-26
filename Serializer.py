import pickle
from ClusterFactory import Cluster
class Serializer:

    @staticmethod
    def save_clusters_list(clusters_list, path):
        with open(path, "w") as f:
            for cluster in clusters_list[:-1]:
                f.write(f"{cluster.get_id()},")
            f.write(f"{clusters_list[-1].get_id()}")

    @staticmethod
    def load_clusters_list(path):
        with open(path) as f:
            line = f.readlines()[0]
            return [int(x) for x in line.split(",")]


    @staticmethod
    def save_dict(d, path):
        with open(path,"wb") as f:
            pickle.dump(d,f)

    @staticmethod
    def load_dict(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def save_clusters(clusters_list, path):
        with open(path, "w") as f:
            for i,cluster in enumerate(clusters_list):
                points = cluster.get_cluster_points()
                for point in points[:-1]:
                    f.write(f"{point}^&^")
                f.write(f"{points[-1]}\n")
    @staticmethod
    def load_clusters(path, embeddings_dict):
        clusters_list  = []
        with open(path) as f:
            for i,line in enumerate(f.readlines()):
                points = [point.replace("\n","") for point in line.split("^&^")]
                try:
                    points_embeddings = [embeddings_dict[point] for point in points]
                except:
                    stop = 1
                clusters_list.append(Cluster(points, points_embeddings, i))
        return clusters_list

