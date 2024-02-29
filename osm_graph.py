import numpy as np
import math
import sys
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as ss
import osmnx as ox
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, save_npz   
from shapely.geometry import Point, LineString


def cal_degree(coord1, coord2):
    x1, y1, x2, y2 = coord1[0], coord1[1], coord2[0], coord2[1]
    deg = math.degrees(math.atan2(y1 - y2, x1 - x2))
    if deg >= 0:
        return deg
    else:
        return 360 + deg
    
def cal_diff(degree1, degree2):
    if degree1 > degree2:
        degree1, degree2 = degree2, degree1
    return min(degree2 - degree1, 360 - degree2 + degree1)

def match(center_coord, coord1, coord2, alpha):
    degree1 = cal_degree(coord1, center_coord)
    degree2 = cal_degree(coord1, coord2)
    degree3 = cal_degree(center_coord, coord2)
    diff1 = cal_diff(degree1, degree2)
    diff2 = cal_diff(degree2, degree3)
    if diff1 <= alpha and diff2 <= alpha:
        return True
    else:
        return False
    

class osm_graph_from_box:
    def __init__(self, north, south, east, west):
        self.G = None
        self.north = north
        self.south = south
        self.east = east
        self.west = west
        
        self.orig_nodes = None
        self.orig_edges = None
        self.orig_num_nodes = None
        self.orig_num_edges = None
        
        self.orig_to_id = None
        self.neigbhor = None

        self.is_merge = False
        
        self.merge_edges = None
        self.merge_num_edges = None

        self.merged_adjacency_matrix = None

    
    def init_graph(self):
        """
        Initialize the osm graph structure using coordinates and obtain node and edge information.
        """
        # Using osmnx to load the data of road
        self.G = ox.graph_from_bbox(
            self.north, 
            self.south, 
            self.east, 
            self.west, 
            network_type='drive'
            )
        # Transfrom G to the gdf format.
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(self.G)
        # Get the geometry information
        geo_edges = gdf_edges[['geometry']].values

        single_edges = set() 

        for i in range(len(geo_edges)):
            geo_edges[i, 0] = tuple(geo_edges[i, 0].coords)

            if geo_edges[i][0][0][0] > geo_edges[i][0][-1][0]:
                geo_edges[i][0] = geo_edges[i][0][::-1]

            single_edges.add(geo_edges[i][0])

        self.orig_edges = list(single_edges)
        # print(self.orig_edges[0])
        # print(self.orig_edges)
        self.orig_num_edges = len(self.orig_edges)
        # print(self.orig_num_edges)

        self.orig_nodes = dict()
        for i in range(self.orig_num_edges):
            road_info = self.orig_edges[i]
            u, v = road_info[0], road_info[-1]
            if u not in self.orig_nodes:
                self.orig_nodes[u] = []
            if v not in self.orig_nodes:
                self.orig_nodes[v] = []
            
            self.orig_nodes[u].append([i, road_info[min(4, len(road_info) - 1)]])
            self.orig_nodes[v].append([i, road_info[max(-4, -len(road_info))]])
        
        self.orig_num_nodes = len(self.orig_nodes)

    def find(self, p, x):
        """
        Union-find algorithm.

        Inputs:
        - p: Union search array.
        - x: The element of the root node to be found.

        Returns::
        - The element ID of the root node.
        """
        if x != p[x]:
            p[x] = self.find(p, p[x])
        return p[x]

    def merge_road(self, alpha = 30):
        """
        Merging the road, which looks like it's straight, using the union-find algorithm.
        After this progress, we can get the new id which is storage in the orig_to_id array.

        Inputs:
        - alpha: An acceptable difference in road angle.
        """
        p = [i for i in range(self.orig_num_edges)]
        
        self.neigbhor = [[] for i in range(self.orig_num_edges)]

        for center_coord, edge_list in self.orig_nodes.items():
            matched = [False for i in range(len(edge_list))]
            for i in range(len(edge_list)):
                if matched[i]: continue
                index1, coord1 = edge_list[i]
                for j in range(i + 1, len(edge_list)):
                    if matched[j]: continue
                    index2, coord2 = edge_list[j]
                    if match(center_coord, coord1, coord2, alpha):
                        p[self.find(p, index1)] = self.find(p, index2)
                        matched[i] = matched[j] = True
                        if index1 == index2:
                            continue
                        self.neigbhor[index1].append(index2)
                        self.neigbhor[index2].append(index1)
                        break

        # Remap ID 
        self.orig_to_id = [0 for i in range(self.orig_num_edges)]
        remap = {}
        self.merge_num_edges = 0
        for i in range(self.orig_num_edges):
            root = self.find(p, i)
            if root not in remap:
                remap[root] = self.merge_num_edges
                self.merge_num_edges += 1
            self.orig_to_id[i] = remap[root]
                
        self.merged_adjacency_matrix = None

        self.is_merge = True
    
    def draw_orig_graph(self):
        """
        Draw the graph which is originally
        """
        return ox.plot_graph(self.G, node_size=10, edge_linewidth=0.5, node_color='green')
    
    def connect_path(self, path1, path2):
        start1 = path1[0]
        end1 = path1[-1]
        start2 = path2[0]
        end2 = path2[-1]
        if end1 == start2:
            return path1 + path2
        if end1 == end2:
            return path1 + path2[::-1]
        if start1 == start2:
            return path1[::-1] + path2
        if start1 == end2:
            return path2 + path1
        print("ERROR")
        return None

    def draw_merge_graph(self):
        """
        Draw the graph which is merged.
        """
        if self.is_merge == False:
            print("Must use merge_road function first.")
            return None
        
        node_set = set()
        nodes_data = {'geometry': []}
        edges_data = {}
        edges_geometry = []
        
        # Connect the road
        connected = [False for i in range(self.orig_num_edges)]
        self.merge_edges = [None for i in range(self.merge_num_edges)]

        for i in range(self.orig_num_edges):
            if connected[i]: continue

            length = len(self.neigbhor[i])

            if length == 0:
                path = self.orig_edges[i]
                connected[i] = True

            elif length == 1:
                path = self.orig_edges[i]
                connected[i] = True
                next = self.neigbhor[i][0]
                while len(self.neigbhor[next]) == 2:
                    path = self.connect_path(path, self.orig_edges[next])
                    connected[next] = True
                    if connected[self.neigbhor[next][0]]:
                        next = self.neigbhor[next][1]
                    else:
                        next = self.neigbhor[next][0]
                path = self.connect_path(path, self.orig_edges[next])
                connected[next] = True
            else:
                continue
                
            # Add the node geometry.
            node_set.add(path[0])
            node_set.add(path[-1])

            edges_data[(i, i, i)] = {}
            edges_geometry.append(LineString(path))

        node_list = list(node_set)
        for node in node_list:
            nodes_data['geometry'].append(Point(node))   
        
        nodes_gdf = gpd.GeoDataFrame(nodes_data, crs="EPSG:4326")
        nodes_gdf["x"] = nodes_gdf["geometry"].x
        nodes_gdf["y"] = nodes_gdf["geometry"].y

        edges_gdf = gpd.GeoDataFrame(list(edges_data.values()), geometry=edges_geometry,
                                 index=pd.MultiIndex.from_tuples(edges_data.keys(), names=['u', 'v', 'k']))

        edges_gdf.crs = "EPSG:4326"

        self.new_G = ox.utils_graph.graph_from_gdfs(nodes_gdf, edges_gdf)

        ox.plot_graph(self.new_G, node_size=10, edge_linewidth=0.5, node_color='green')


    def get_merged_adj_matrix(self):
        """
        Get the new adjacency matrix after merging the road.
        """
        if self.is_merge == False:
            print("Must use merge_road function first.")
            return None

        if self.merged_adjacency_matrix != None:
            return self.merged_adjacency_matrix
        
        data = []
        row_indices = []
        col_indices = []
        check = set()

        for center_coord, edge_list in self.orig_nodes.items():
            for i in range(len(edge_list)):
                edge_id1 = self.orig_to_id[edge_list[i][0]]
                for j in range(i + 1, len(edge_list)):
                    edge_id2 = self.orig_to_id[edge_list[j][0]]
                    if edge_id1 != edge_id2:
                        if (edge_id1, edge_id2) in check:
                            continue
                        else:
                            check.add((edge_id1, edge_id2))
                            check.add((edge_id2, edge_id1))
                        data.append(1)
                        row_indices.append(edge_id1)
                        col_indices.append(edge_id2)
                        data.append(1)
                        row_indices.append(edge_id2)
                        col_indices.append(edge_id1)
        
        data = np.array(data)
        row_indices = np.array(row_indices)
        col_indices = np.array(col_indices)

        self.merged_adjacency_matrix = csr_matrix((data, (row_indices, col_indices)), 
                                                  shape=(self.merge_num_edges, self.merge_num_edges))
        
        return self.merged_adjacency_matrix

    def get_similarity_of_adj_matrix(self):
        """
        Get the node similarity between each row in the adjacency_matrix.
        """

        if self.merged_adjacency_matrix == None:
            print("Must use get_merged_adj_matrix function.")
            return None
        
        cosine_similarities = cosine_similarity(self.merged_adjacency_matrix, dense_output=False)

        return cosine_similarities
        




        




            




