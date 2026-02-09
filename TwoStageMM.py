import copy
import pdb
import os
import glob
import re
import random
import time
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from datetime import datetime
from collections import defaultdict


class MinFleet(object):
    def __init__(self):
        self.stand_set = ["S1", "S2", "S3", "S4", "N5", "N6", "N7", "N8", "N9", "N10", "S11", "N12", "R13", "R14",
                          "R15", "R16", "R17", "R18", "R19", "R20", "R21", "S23", "N24", "S25", "N26", "S27", "N28",
                          "S29", "N30", "S31", "N32", "S33", "N34", "S35", "N36", "W40", "S41", "W42", "S43", "W44",
                          "S45", "W46", "S47", "W48", "S49", "W50", "N60", "W61", "N62", "W63", "N64", "W65", "N66",
                          "W67", "N68", "W69", "N70", "W71"]  # 58
        self.station_set_total = [2, 3, 4, 7, 12, 15, 16, 17, 20, 21, 25, 27, 30, 33, 34, 39, 43, 44, 47, 48, 51,
                                  54, 56]  # 23

        distance_file = 'distance_matrix.csv'  # distance matrix
        ev_file_path = 'AED_data.csv'   # AE-Dolly file
        ev_df = pd.read_csv(ev_file_path, sep=',')

        self.tra_di_df = pd.read_csv(distance_file, header=None)
        self.tra_speed = ev_df.iloc[3, 2]  # m/s
        self.tra_ti_df = self.tra_di_df / self.tra_speed
        self.pow_rate = ev_df.iloc[7, 2]   # %/m
        self.charging_rate = ev_df.iloc[6, 2]   # %/s
        self.tra_pow_df = self.tra_di_df * self.pow_rate

        self.alpha = 300
        self.duration = 80      # longest path length

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

    def node_pow_cons(self, energy_para):
        node_cons = []
        for task in self.flight_node_set:
            node_cons.append(task[4] * energy_para)
        return node_cons

    def find_in_2d_list(self, target, list_of_lists):
        for i, sublist in enumerate(list_of_lists):
            if target in sublist:
                return i
        return -1

    def min_path_cov(self, g, node_cons):
        temp_graph = nx.DiGraph()
        temp_graph.add_nodes_from(g.nodes)
        temp_graph_edges = []
        paths = []
        lengths = []
        for component in nx.weakly_connected_components(g):
            if len(component) != 1:
                sub_graph = g.subgraph(component)
                bigraph = nx.DiGraph()  # construct bipartite graph
                top_nodes = [i for i in component]
                bottom_nodes = [i + len(self.flight_node_set) for i in top_nodes]
                bigraph.add_nodes_from(top_nodes, bipartite=0)
                bigraph.add_nodes_from(bottom_nodes, bipartite=1)
                for i in top_nodes:
                    for j in top_nodes:
                        if (i, j) in sub_graph.edges:
                            bigraph.add_edge(i, j + len(self.flight_node_set), weight=sub_graph.get_edge_data(i, j)['weight'])
                for bi_comp in list(nx.weakly_connected_components(bigraph)):
                    bigraph_sub = bigraph.subgraph(bi_comp)
                    if len(bi_comp) != 1:
                        match = nx.bipartite.maximum_matching(bigraph_sub)  # unweighted matching
                        # dic_match =  # weighted matching
                        for start_point, end_point in match.items():
                            if start_point >= len(self.flight_node_set):
                                start_point = start_point - len(self.flight_node_set)
                            if end_point >= len(self.flight_node_set):
                                end_point = end_point - len(self.flight_node_set)
                            temp_graph_edges.append((start_point, end_point))
        temp_graph.add_edges_from(temp_graph_edges)
        for component in nx.weakly_connected_components(temp_graph):
            paths.append(list(component))
        sorted_paths = []
        for path in paths:
            sorted_paths.append(sorted(path, key=lambda item: self.flight_node_set[item][1]))
        for path in sorted_paths:
            current_length = node_cons[path[0]]
            for i in range(len(path)):
                if i != len(path)-1:
                    current_length = (current_length +
                                      self.tra_pow_df.iloc[int(self.flight_node_set[path[i]][2]), int(
                                          self.flight_node_set[path[i + 1]][0])] +
                                      node_cons[path[i + 1]])
            lengths.append(current_length)
        return sorted_paths, lengths

    def cutting_algorithm(self, g, paths, ind, len_val, node_cons):
        cut_left, cut_right = 0, 0
        path = copy.deepcopy(paths[int(ind)])
        #######################################################################################################
        """
        # source cut
        cut_length = self.duration
        current_length = node_cons[path[-1]]
        for i in range(len(path)):
            if i <= len(path) - 2:
                current_length = (current_length +
                                  self.tra_pow_df.iloc[int(self.flight_node_set[path[-i - 2]][2]), int(
                                      self.flight_node_set[path[-i - 1]][0])] +
                                  node_cons[path[-i - 2]])
                if current_length >= cut_length:
                    cut_left = path[-i-2]
                    cut_right = path[-i-1]
                    break
        """
        ###########################################################################################################
        """
        # mid cut
        cut_length = len_val / 2
        current_length = node_cons[path[0]]
        for i in range(len(path)):
            if i != len(path) - 1:
                current_length = (current_length +
                                  self.tra_pow_df.iloc[int(self.flight_node_set[path[i]][2]), int(
                                      self.flight_node_set[path[i + 1]][0])] +
                                  node_cons[path[i + 1]])
                if current_length >= cut_length:
                    cut_left = path[i - 1]
                    cut_right = path[i]
                    break
        """
        #######################################################################################################

        # sink cut
        cut_length = self.duration
        current_length = node_cons[path[0]]
        for i in range(len(path)):
            if i >= 1:
                current_length = (current_length +
                                  self.tra_pow_df.iloc[int(self.flight_node_set[path[i - 1]][2]), int(
                                      self.flight_node_set[path[i]][0])] +
                                  node_cons[path[i]])
                if current_length >= cut_length:
                    cut_left = path[i-1]
                    cut_right = path[i]
                    break

        #######################################################################################################
        cut_set = g.out_edges(cut_left)
        g.remove_edges_from(list(cut_set))
        return g

    def second_layer_construction(self, g, data, paths, energy_para):
        second_layer = g.copy()
        #  node convergence
        for i in range(len(paths)):
            startpoint = paths[i][0]
            endpoint = paths[i][0]
            # calculate the energy consumption
            energy_consumption = self.flight_node_set[paths[i][0]][4]
            for j in range(len(paths[i])):
                if j != 0:
                    energy_consumption = (energy_consumption +
                                          self.tra_pow_df.iloc[int(self.flight_node_set[paths[i][j-1]][2]), int(self.flight_node_set[paths[i][j]][0])] +
                                          self.flight_node_set[paths[i][j]][4])
            for j in paths[i]:
                if self.flight_node_set[j][1] < self.flight_node_set[startpoint][1]:
                    startpoint = j
                if self.flight_node_set[j][3] > self.flight_node_set[endpoint][3]:
                    endpoint = j
            self.flight_node_set.append([self.flight_node_set[startpoint][0], self.flight_node_set[startpoint][1],
                                         self.flight_node_set[endpoint][2], self.flight_node_set[endpoint][3],
                                         energy_consumption * energy_para])
            second_layer.add_node(i + len(data.values[0:].tolist()), weight=energy_consumption)
            second_layer.remove_nodes_from(paths[i])
        second_layer.clear_edges()
        stations = self.station_set_total
        for i in second_layer.nodes():
            for j in second_layer.nodes():
                start_node = self.flight_node_set[i][2]
                end_node = self.flight_node_set[j][0]
                tra_ti_df = self.tra_ti_df
                temp = min(tra_ti_df[start_node][station] + tra_ti_df[station][end_node] for station in stations)
                if self.flight_node_set[j][1] - self.flight_node_set[i][3] >= temp + self.flight_node_set[j][4] / self.charging_rate:
                    second_layer.add_edge(i, j)
        return second_layer

    def network_instance_generation(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                if len(df) > 1:
                    df = df.iloc[1:-1]
                df.to_csv(file_path, index=False)

    def read_csv_if_not_empty(self, file_path):
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            # print(f"File {file_path} does not exist or is empty.")
            return None
        try:
            df = pd.read_csv(file_path, header=None if os.path.getsize(file_path) == 0 else 0)
            if df.empty:
                # print(f"File {file_path} is empty (contains no rows).")
                return None
            return df
        except Exception as e:
            # print(f"Error reading file {file_path}: {e}")
            return None


if __name__ == '__main__':
    start_time = time.perf_counter()
    fleet = []

    folder_path = 'instances'  # folder path of instances
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    sorted_files = sorted(csv_files)
    ener_para = 1
    alp = 300

    for file in sorted_files:
        sce = MinFleet()
        sce.alpha = alp
        match = re.search(r'\d{4}-\d{2}-\d{2}', file)
        date = match.group()
        if date:
            print("current date: ", date)
            df = sce.read_csv_if_not_empty(file)
            # df = pd.read_csv(2023-02-27.csv')
            if df is not None:
                sce.flight_node_set = df.values[0:].tolist()

                # initial network
                sce.adj_generate()
                initial_graph = sce.digraph_generate()   # initial directed task network generation
                node_consumption = sce.node_pow_cons(ener_para)
                node_cons_dict = {index: value for index, value in enumerate(node_consumption)}
                nx.set_node_attributes(initial_graph, node_cons_dict, "weight")
                initial_network_time = time.perf_counter()

                # graph cutting
                path_cover, path_length = sce.min_path_cov(initial_graph, node_consumption)
                cutting_graph = initial_graph.copy()
                if path_length:
                    while max(path_length) >= sce.duration:
                        len_value = max(path_length)
                        index = path_length.index(len_value)
                        cutting_graph = sce.cutting_algorithm(initial_graph, path_cover, index, len_value, node_consumption)
                        path_cover, path_length = sce.min_path_cov(initial_graph, node_consumption)
                path_cover_time = time.perf_counter()

                # construct second layer network
                second_digraph = sce.second_layer_construction(cutting_graph, df, path_cover, ener_para)
                final_graph = nx.DiGraph()
                final_graph.add_nodes_from(second_digraph.nodes())
                final_graph_edges = []
                final_paths = []
                final_lengths = []
                second_bigraph = nx.DiGraph()
                first_set = [i for i in second_digraph.nodes]
                second_set = [i + len(sce.flight_node_set) for i in first_set]
                second_bigraph.add_nodes_from(first_set, bipartite=0)
                second_bigraph.add_nodes_from(second_set, bipartite=1)
                for i in first_set:
                    for j in first_set:
                        if (i, j) in second_digraph.edges:
                            second_bigraph.add_edge(i, j + len(sce.flight_node_set))
                match_number = 0
                total_match = {}
                for bi_component in list(nx.weakly_connected_components(second_bigraph)):
                    sub_bigraph = second_bigraph.subgraph(bi_component)
                    if len(bi_component) != 1:
                        dic_match = nx.bipartite.maximum_matching(sub_bigraph)  # unweighted matching
                        total_match.update(dic_match)
                        match_number = match_number + len(dic_match)

                for start_point, end_point in total_match.items():
                    if start_point >= len(sce.flight_node_set):
                        start_point = start_point - len(sce.flight_node_set)
                    if end_point >= len(sce.flight_node_set):
                        end_point = end_point - len(sce.flight_node_set)
                    final_graph_edges.append((start_point, end_point))
                final_graph.add_edges_from(final_graph_edges)
                for component in nx.weakly_connected_components(final_graph):
                    final_paths.append(list(component))

                sorted_final_paths = []
                for path in final_paths:
                    sorted_final_paths.append(sorted(path, key=lambda item: sce.flight_node_set[item][1]))
                for path in sorted_final_paths:
                    current_length = sce.flight_node_set[path[0]][4]
                    for i in range(len(path)):
                        if i != len(path) - 1:
                            current_length = (current_length +
                                              sce.tra_pow_df.iloc[int(sce.flight_node_set[path[i]][2]), int(
                                              sce.flight_node_set[path[i + 1]][0])] + sce.flight_node_set[path[i+1]][4])
                    final_lengths.append(current_length)
                # print(final_lengths)
                # pdb.set_trace()

                second_layer_min_fleet_time = time.perf_counter()
                sce.min_fle_size = second_digraph.number_of_nodes() - match_number/2
                print("fleet size = ", sce.min_fle_size)
                # print("running time = ", second_layer_min_fleet_time - start_time)
                fleet.append(sce.min_fle_size)
    print(fleet)
