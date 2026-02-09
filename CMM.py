import copy
import pdb
import os
import random
import time
import glob
import re
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from datetime import datetime


class MinFleet(object):
    def __init__(self, distance_file, vehicle_file):
        self.stand_set = ["S1", "S2", "S3", "S4", "N5", "N6", "N7", "N8", "N9", "N10", "S11", "N12", "R13", "R14",
                          "R15", "R16", "R17", "R18", "R19", "R20", "R21", "S23", "N24", "S25", "N26", "S27", "N28",
                          "S29", "N30", "S31", "N32", "S33", "N34", "S35", "N36", "W40", "S41", "W42", "S43", "W44",
                          "S45", "W46", "S47", "W48", "S49", "W50", "N60", "W61", "N62", "W63", "N64", "W65", "N66",
                          "W67", "N68", "W69", "N70", "W71"]

        self.tra_di_df = pd.read_csv(distance_file, header=None)
        self.tra_speed = pd.read_csv(vehicle_file, sep=',').iloc[3, 2]  # m/s
        self.tra_ti_df = pd.read_csv(distance_file, header=None) / self.tra_speed
        self.pow_rate = pd.read_csv(vehicle_file, sep=',').iloc[7, 2]   # %/m
        self.charging_rate = pd.read_csv(vehicle_file, sep=',').iloc[6, 2]   # %/s
        self.tra_pow_df = pd.read_csv(distance_file, header=None) * self.pow_rate

        self.alpha = 3600 * 24    # maximum waiting time
        self.duration = 80      # longest path length %/maximum power
        self.charging_time = 3600

        self.flight_node_set = []
        self.task_adj_matrix = []
        self.min_fle_size = 0

    def adj_generate(self):
        for i in range(len(self.flight_node_set)):
            temp_list = []
            for j in range(len(self.flight_node_set)):
                if (self.alpha >= self.flight_node_set[j][1] - self.flight_node_set[i][3] >=
                        self.tra_ti_df[self.flight_node_set[i][2]][self.flight_node_set[j][0]]):
                    temp_list.append(1)
                else:
                    temp_list.append(0)
            self.task_adj_matrix .append(temp_list)

    def digraph_generate(self):
        # direct graph
        g = nx.DiGraph()
        g.add_nodes_from(range(len(self.flight_node_set)))
        for i in g.nodes:
            for j in g.nodes:
                if self.task_adj_matrix[i][j] == 1:
                    g.add_edge(i, j, weight=self.tra_pow_df.iloc[int(self.flight_node_set[i][2]), int(self.flight_node_set[j][0])])
        return g

    def node_pow_cons(self):
        node_cons = []
        for task in self.flight_node_set:
            node_cons.append(task[4])
        return node_cons

    def find_in_2d_list(self, target, list_of_lists):
        for i, sublist in enumerate(list_of_lists):
            if target in sublist:
                return i
        return -1

    def read_csv_if_not_empty(self, file_path):
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"File {file_path} does not exist or is empty.")
            return None
        try:
            df = pd.read_csv(file_path, header=None if os.path.getsize(file_path) == 0 else 0)
            if df.empty:
                print(f"File {file_path} is empty (contains no rows).")
                return None
            return df
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None


if __name__ == '__main__':
    start_time = time.perf_counter()

    distance_file_path = 'distance_matrix.csv'  # distance matrix
    vehicle_file_path = 'AED_data.csv'   # AE-Dolly file
    df = pd.read_csv('2023-02-27.csv')   # instance file

    fleet = []
    unconstrained_fleet = []
    task_number = []
    folder_path = 'instance-revision'   # folder path of instances
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    sorted_files = sorted(csv_files)

    for file in sorted_files:
        sce = MinFleet(distance_file_path, vehicle_file_path)
        match = re.search(r'\d{4}-\d{2}-\d{2}', file)
        date = match.group()
        if date:
            print("current date: ", date)
            df = sce.read_csv_if_not_empty(file)
            if df is not None:
                sce.flight_node_set = df.values[0:].tolist()
                task_number.append(len(sce.flight_node_set))

                # initial network
                sce.adj_generate()
                initial_graph = sce.digraph_generate()   # initial directed task network generation
                node_consumption = sce.node_pow_cons()
                node_cons_dict = {index: value for index, value in enumerate(node_consumption)}
                nx.set_node_attributes(initial_graph, node_cons_dict, "weight")
                initial_network_time = time.perf_counter()

                # bipartite graph
                bigraph = nx.DiGraph()  # construct bipartite graph
                top_nodes = [i for i in range(len(sce.flight_node_set))]
                bottom_nodes = [i + len(sce.flight_node_set) for i in top_nodes]
                bigraph.add_nodes_from(top_nodes, bipartite=0)
                bigraph.add_nodes_from(bottom_nodes, bipartite=1)
                for i in top_nodes:
                    for j in top_nodes:
                        if (i, j) in initial_graph.edges:
                            bigraph.add_edge(i, j + len(sce.flight_node_set), weight=initial_graph.get_edge_data(i, j)['weight'])

                # maximum matching
                max_matching = []
                match_number = 0
                for component in nx.weakly_connected_components(bigraph):
                    if len(component) != 1:
                        bigraph_sub = bigraph.subgraph(component)
                        match = nx.bipartite.maximum_matching(bigraph_sub)  # unweighted matching
                        match_number = match_number + len(match)
                        new_match = {}
                        for k, v, idx in zip(match.keys(), match.values(), range(len(match))):
                            if idx < len(match) / 2:
                                new_match[k] = v
                        max_matching.append(list(new_match.items()))
                sce.min_fle_size = len(sce.flight_node_set) - match_number / 2
                print("unconstrained fleet size = ", sce.min_fle_size)
                unconstrained_fleet.append(sce.min_fle_size)
                # print(max_matching)
                # pdb.set_trace()

                # bipartite graph cutting
                for match in max_matching:
                    # print(match)
                    while match:
                        start_point = match[0][0]
                        end_point = match[0][1] - len(sce.flight_node_set)
                        edges_to_remove = list(bigraph.out_edges(start_point, data=False))
                        bigraph.remove_edges_from(bigraph.out_edges(edges_to_remove))
                        bigraph.add_edge(match[0][0], match[0][1])
                        current_length = node_consumption[start_point] + sce.tra_pow_df.iloc[int(sce.flight_node_set[start_point][2]), int(
                                                      sce.flight_node_set[end_point][0])] + node_consumption[end_point]
                        match.remove((start_point, end_point + len(sce.flight_node_set)))
                        search_index = 0
                        cut_left = 0
                        cut_right = 0
                        while current_length <= sce.duration and search_index == 0:
                            search_index = 1
                            for (key, value) in match:
                                if key == end_point:
                                    search_index = 0
                                    end_point = value - len(sce.flight_node_set)
                                    current_length = current_length + sce.tra_pow_df.iloc[int(sce.flight_node_set[key][2]), int(
                                                     sce.flight_node_set[end_point][0])] + node_consumption[end_point]
                                    # print(current_length)
                                    if current_length >= sce.duration:
                                        end_point = key
                                        edges_to_remove = []
                                        for (i, j) in bigraph.out_edges(key):
                                            if sce.flight_node_set[i][3] + sce.charging_time >= \
                                                    sce.flight_node_set[j - len(sce.flight_node_set)][1]:
                                                edges_to_remove.append((i, j))
                                        # print(edges_to_remove)
                                        # pdb.set_trace()
                                        bigraph.remove_edges_from(edges_to_remove)
                                        match.remove((key, value))
                                        break
                                    edges_to_remove = list(bigraph.out_edges(key, data=False))
                                    # print(edges_to_remove)
                                    bigraph.remove_edges_from(edges_to_remove)
                                    bigraph.add_edge(key, value)
                                    match.remove((key, value))

                # fleet size
                match_number = 0
                for bi_component in list(nx.weakly_connected_components(bigraph)):
                    sub_bigraph = bigraph.subgraph(bi_component)
                    if len(bi_component) != 1:
                        dic_match = nx.bipartite.maximum_matching(sub_bigraph)  # unweighted matching
                        match_number = match_number + len(dic_match)
                sce.min_fle_size = len(sce.flight_node_set) - match_number / 2
                finish_time = time.perf_counter()
                print("constrained fleet size = ", sce.min_fle_size)
                print("running time = ", finish_time - start_time)
                fleet.append(sce.min_fle_size)
    print(task_number)
    print(unconstrained_fleet)
    print(fleet)
