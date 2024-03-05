from osm_graph import osm_graph_from_box
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, save_npz


if __name__ == '__main__':
    # north, south, east, west = 22.573, 22.518, 114.1107, 114.024
    north, south, east, west = 34.60, 34.00, 108.53, 109.66  # 西安
    graph = osm_graph_from_box(north, south, east, west)
    graph.init_graph()
    # graph.draw_orig_graph()
    graph.merge_road()
    graph.draw_merge_graph()
    # matrix = graph.get_merged_adj_matrix()

    # save_npz('sparse_matrix.npz', matrix)
    # print(matrix.shape)

    # c = graph.get_similarity_of_adj_matrix()
