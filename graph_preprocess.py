from osm_graph import osm_graph_from_box
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


if __name__ == '__main__':
    north, south, east, west = 22.573, 22.518, 114.1107, 114.024
    graph = osm_graph_from_box(north, south, east, west)
    graph.init_graph()
    # t = graph.draw_orig_graph()
    graph.merge_road()
    # graph.draw_merge_graph()
    graph.get_merged_adj_matrix()

    c = graph.get_similarity_of_adj_matrix()
    print(c)
