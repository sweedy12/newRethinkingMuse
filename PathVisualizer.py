
class PathFinder:
    def __init__(self, node_to_neighbors, top_order, node_to_label):
        self.node_to_neighbors=  node_to_neighbors
        self.top_order = top_order
        self.node_to_label=  node_to_label

    def find_paths_in_graph(self, num_paths):
        import random
        paths = []
        starting_points = []
        seen_points = set()
        for node in self.top_order:
            if node not in seen_points:
                starting_points.append(node)
            for neighbor in self.node_to_neighbors[node]:
                seen_points.add(neighbor)
        random.shuffle(starting_points)
        if len(starting_points) >= num_paths:
            starting_points = starting_points[:num_paths]
        for starting_point in starting_points:
            paths.append(self.unfold_path(starting_point))
        return paths


    def unfold_path(self, starting_node):
        import random
        path = [starting_node]
        cur_neighbors = self.node_to_neighbors[starting_node]
        while cur_neighbors:
            rand_neighbor = random.choice(cur_neighbors)
            path.append(rand_neighbor)
            cur_neighbors = self.node_to_neighbors[rand_neighbor]
        return path


class PathVisualizer:

    @staticmethod
    def get_path_str(path):
        str = ""
        for cluster in path[:-1]:
            str += f"{cluster.get_random_point()} -------> \n"
        str += f"{path[-1].get_random_point()}\n"
        return str

    @staticmethod
    def write_paths_to_file(fname, paths):
        with open(fname, "w") as f:
            f.write(f"Examples of random paths from the graph\n\n")
            f.write("--------------------------------------------")
            for i,path in enumerate(paths):
                f.write(f"Path number {i+1}:\n")
                f.write(PathVisualizer.get_path_str(path))
                f.write("----------------------------------------------------")
    