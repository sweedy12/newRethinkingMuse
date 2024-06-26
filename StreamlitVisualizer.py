import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import os
from ClusterVisualizer import ClusterVisualizer
from ClusterFactory import Cluster

def create_toy_pyvis_graph():
    net = Network(height="600px", width="100%", font_color="black", heading='toy example', directed = True)
    net.barnes_hut()
    nodes = ["nir", "nir2", "nir3", "nir4"]
    nodes_str = ["this is how we nir it", "this one is a bit shorter", "i am nir", "sdfjoisdfjsjdfiojdsi sdijoidsjf sdiofjoisdf dsifjodsi "]
    node_to_level = {"nir":0, "nir2":1, "nir3":1, "nir4":2}
    level_to_color = {0:"blue", 1:"green", 2: "red"}
    for i,node in enumerate(nodes):
        net.add_node(n_id = node, label = nodes_str[i], color=level_to_color[node_to_level[node]])

    for i, node in enumerate(nodes):
        for node2 in nodes:
            if node != node2:
                net.add_edge(node, node2, arrowStrikethrough = False)
    net.show("colored_toy_example.html", notebook= False)


class StreamlitVisualizer:

    @staticmethod
    def graph_with_options_trial(dir_path):
        st.title("Interactive graph demo")
        listdir  = os.listdir(dir_path)
        options = []
        for x in listdir:
            options.append(x)
        st.sidebar.header("Graph options")
        selectbox = st.sidebar.selectbox("Choose a graph to display", options)
        HtmlFile = open(f"{dir_path}/{selectbox}", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=1200, width=1000)




    @staticmethod
    def graph_trial(graph_path):
        HtmlFile = open(graph_path, 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=1200, width=1000)



pyvis_path = "cluster_visualization/pyvis_visulizations/pyvis_toy.html"
dir_path = "C:\\Users\\sweed\\PycharmProjects\\RethinkingMuse\\cluster_visualization\\3004\\pyvis_clusters\\200"
nodes = [Cluster([f"nir_{str(i)}"], None, i) for i in range(300)]


# node_to_neighbors = {nodes[i] : [nodes[i+1]] for i in range(299)}
# node_to_neighbors[nodes[-1]]= []
# node_to_level = {nodes[i]:i for i in range(300)}
# ClusterVisualizer.draw_colored_cluster_graph_with_pyvis(nodes, node_to_neighbors, node_to_level,"wallak300.html")
#
# # create_toy_pyvis_graph()
StreamlitVisualizer.graph_with_options_trial(dir_path)