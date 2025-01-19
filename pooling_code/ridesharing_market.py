"""
This python script is designed to quantify efficiency for pooling service of
ridesharing market caused by market segmentation of competitors, Uber and Lyft for instance.

@author: Xiaotong Guo
"""


import csv
import sys
import numpy as np
import networkx as nx
import pickle
import random
import utm
import matplotlib.pyplot as plt
import pandas as pd
import time
import multiprocessing
import os
import itertools


def main():
    """
    Main function for quantifying ridesharing market efficiency loss
    """

    random.seed(10)

    parallel_groups = 40
    weight = "distance"

    sharing_threshold = 180
    print("The tightness factor is: " + str(sharing_threshold))
    EL = Efficiency_loss(sharing_threshold=sharing_threshold)

    scaling_factor = 1
    print("The thickness factor is: " + str(scaling_factor))

    graph, total_distance = EL.graph_sampling(scaling_factor)
    print("Number of nodes in graph: " + str(len(graph.nodes)))
    print("Total distance is : " + str(total_distance))

    critical_number = int((len(graph.nodes) / parallel_groups))
    print("Critical number is: " + str(critical_number))

    matching_results = EL.heuristic_max_weight_matching(critical_number=critical_number, G1=graph, weight=weight)
    total_distance_saving = 0
    for edge in matching_results:
        distance_saving = EL.graph[edge[0]][edge[1]][weight]
        total_distance_saving += distance_saving
    print("Saving is: " + str(total_distance_saving))
    print("Saving caused by sharing: " + str(total_distance_saving / total_distance))

    sub_graph1, sub_graph2 = EL.graph_geo_split(graph)
    for i in range(11):
        randomness = i * 0.1
        output_randomness = 1 - randomness
        print("The randomness factor is: " + str(output_randomness))
        # Draw % trips at each sub graph, and draw the remaining % trips totally random
        node_list1 = list(sub_graph1.nodes)
        number_nodes_subgraph1 = int(len(node_list1) * randomness)
        sub_node_list1 = random.sample(node_list1, k=number_nodes_subgraph1)
        remain_node_list1 = list(set(node_list1) - set(sub_node_list1))

        node_list2 = list(sub_graph2.nodes)
        number_nodes_subgraph2 = int(len(node_list2) * randomness)
        sub_node_list2 = random.sample(node_list2, k=number_nodes_subgraph2)
        remain_node_list2 = list(set(node_list2) - set(sub_node_list2))

        remain_node_list = remain_node_list1 + remain_node_list2
        remain_node_list_1 = random.sample(remain_node_list, k=int(len(remain_node_list) * 0.5))
        remain_node_list_2 = list(set(remain_node_list) - set(remain_node_list1))

        node_list_1 = sub_node_list1 + remain_node_list_1
        node_list_2 = sub_node_list2 + remain_node_list_2

        sub_graph_1 = graph.subgraph(node_list_1).copy()
        sub_graph_2 = graph.subgraph(node_list_2).copy()

        split_matching_results = EL.heuristic_max_weight_matching(critical_number=critical_number,
                                                                  G1=sub_graph_1, G2=sub_graph_2, weight=weight, )

        split_distance_saving = 0
        for edge in split_matching_results:
            distance_saving = graph[edge[0]][edge[1]][weight]
            split_distance_saving += distance_saving

        print("The distance saving after market segmentation is: " + str(split_distance_saving))
        print("Efficiency loss is: " + str(total_distance_saving - split_distance_saving))


    # VMT2_results = []
    # VMT3_results = []
    #
    # for i in range(10):
    #     start_time = time.time()
    #     # sharing_threshold = (4+i) * 100
    #     sharing_threshold = (i+1) * 30
    #     print("The tightness factor is: " + str(sharing_threshold))
    #     EL = Efficiency_loss(sharing_threshold=sharing_threshold)
    #
    #     for j in range(1):
    #         #scaling_factor = (j + 1) * 0.1
    #         scaling_factor = 1
    #         print("The thickness factor is: " + str(scaling_factor))
    #
    #         graph, total_distance = EL.graph_sampling(scaling_factor)
    #         print("Number of nodes in graph: " + str(len(graph.nodes)))
    #         print("Total distance is : " + str(total_distance))
    #
    #         critical_number = int((len(graph.nodes) / parallel_groups))
    #         print("Critical number is: " + str(critical_number))
    #
    #         matching_results = EL.heuristic_max_weight_matching(critical_number=critical_number, G1=graph, weight=weight)
    #         #matching_results = EL.greedy_max_weight_matching(graph=graph,weight=weight)
    #         edge_weight_list = []
    #         total_distance_saving = 0
    #         for edge in matching_results:
    #             distance_saving = EL.graph[edge[0]][edge[1]][weight]
    #             edge_weight_list.append(EL.graph[edge[0]][edge[1]]['delta'])
    #             total_distance_saving += distance_saving
    #         print("Saving is: " + str(total_distance_saving))
    #         print("Saving caused by sharing: " + str(total_distance_saving / total_distance))
    #
    #         avg_weight = np.mean(edge_weight_list)
    #         matching_rate = round((len(matching_results) * 2 / len(graph.nodes)) * 100, 2)
    #         VMT2_results.append([sharing_threshold, avg_weight, matching_rate])
    #
    #         for k in range(1):
    #             split_factor = 0.5
    #             # split_factor = (k + 1) * 0.05
    #             print("The unbalancedness factor is: " + str(split_factor))
    #             graph1, graph2 = EL.graph_random_split(split_factor, graph=graph)
    #             split_matching_results = EL.heuristic_max_weight_matching(critical_number=critical_number, G1=graph1, G2=graph2, weight=weight)
    #             #split_matching_results = EL.greedy_max_weight_matching(graph=graph1,weight=weight) \
    #             #                        + EL.greedy_max_weight_matching(graph=graph2,weight=weight)
    #             edge_weight_list = []
    #             split_distance_saving = 0
    #             for edge in split_matching_results:
    #                 distance_saving = graph[edge[0]][edge[1]][weight]
    #                 edge_weight_list.append(EL.graph[edge[0]][edge[1]]['delta'])
    #                 split_distance_saving += distance_saving
    #             print("The distance saving after market segmentation is: " + str(split_distance_saving))
    #             print("Efficiency loss is: " + str(total_distance_saving - split_distance_saving))
    #
    #             avg_weight = np.mean(edge_weight_list)
    #             matching_rate = round((len(split_matching_results) * 2 / len(graph.nodes)) * 100, 2)
    #             VMT3_results.append([sharing_threshold, avg_weight, matching_rate])
    #
    #             print("--------------------")
    #         print("-------------------------------")
    #
    #     end_time = time.time()
    #     print("This iteration costs time: " + str(end_time - start_time))
    #     print("----------------------------------------------")
    #
    # print(VMT2_results)
    # print(VMT3_results)
    # pd.DataFrame(VMT2_results).to_csv('VMT2.csv', header=False, index=False)
    # pd.DataFrame(VMT3_results).to_csv('VMT3.csv', header=False, index=False)




class Efficiency_loss(object):

    def __init__(self, sharing_threshold):

        self.pool_pricing_distance = 0.8
        self.pool_pricing_time = 0.2
        self.x_pricing_distance = 0.8
        self.x_pricing_time = 0.28
        self.booking_fare = 2.3
        self.min_price = 5.8

        self.trip_info = {}  # Key: trip_id, Value: [time,from,to,duration,length]
        with open('data/trips-114.csv') as g:
            wt = csv.reader(g, delimiter=',')
            for row in wt:
                self.trip_info[int(row[0])] = row[1:]

        self.raw_data = {}
        with open('data/data.csv') as g:
            wt = csv.reader(g, delimiter=',')
            for row in wt:
                self.raw_data[int(row[0])] = row[1:]

        self.graph = self.generate_graph(sharing_threshold)
        #pickle.dump(self.graph, open("Graph.p", "wb"))
        #self.graph = pickle.load(open("Graph.p", "rb"))

    def generate_graph(self, sharing_threshold):
        """
        Function to generate graph based on data
        :return: graph in networkx format
        """
        node_list = []
        for node in self.trip_info:
            if node == -1:
                continue
            node_list.append(node)
        node_list = list(set(node_list))
        node_list.sort()

        G = nx.Graph()
        for node in node_list:
            G.add_node(node)
            try:
                information = self.raw_data[node]
            except KeyError:
                continue
            index = int(len(self.raw_data[node]) / 4)
            node_distance = int(self.trip_info[node][4])
            node_time = int(self.trip_info[node][3])
            for i in range(index):
                node_b = node + int(information[i*4])
                delta = int(information[i*4 + 1]) * 10
                if delta > sharing_threshold:
                    continue
                node_b_distance = int(self.trip_info[node_b][4])
                node_b_time = int(self.trip_info[node_b][3])
                travel_time_save = int(information[i*4 + 2])
                travel_distance_save = int(information[i*4 + 3])
                node_price = self.booking_fare + node_distance/1609.34 * self.pool_pricing_distance \
                             + node_time/60 * self.pool_pricing_time
                node_b_price = self.booking_fare + node_b_distance/1609.34 * self.pool_pricing_distance \
                               + node_b_time/60 * self.pool_pricing_time
                if node_price < self.min_price:
                    node_price = self.min_price
                if node_b_price < self.min_price:
                    node_b_price = self.min_price
                profit = node_price + node_b_price - self.booking_fare \
                         - (node_distance + node_b_distance - travel_distance_save)/1609.34 * self.x_pricing_distance \
                         - (node_time + node_b_time - travel_time_save)/60 * self.x_pricing_time
                G.add_edge(node, node_b, time=travel_time_save, distance=travel_distance_save, profit=profit, delta=delta)

        return G

    def heuristic_max_weight_matching(self, critical_number, G1, G2 = None, weight = None):
        """
        Function to do the heuristic maximum weight matching
        :param G: graph in networkx format
        :param weight: attributes to maximize through matching
        :return:
        """
        graph_list = []

        node_list = list(G1.nodes)
        node_list.sort()
        list_length = len(node_list)
        number_of_groups = int(list_length / critical_number)
        for i in range(number_of_groups):
            if i == 0:
                sub_node_list = node_list[:int(list_length/number_of_groups)]
            elif i == number_of_groups - 1:
                sub_node_list = node_list[int(list_length / number_of_groups)*i:]
            else:
                sub_node_list = node_list[int(list_length/number_of_groups)*i:int(list_length/number_of_groups)*(i+1)]
            #sub_graph = self.generate_sub_graph(sub_node_list)
            sub_graph = G1.subgraph(sub_node_list).copy()
            graph_list.append(sub_graph)

        if G2 != None:
            node_list2 = list(G2.nodes)
            node_list2.sort()
            list_length2 = len(node_list2)
            number_of_groups2 = int(list_length2 / critical_number)
            for i in range(number_of_groups2):
                if i == 0:
                    sub_node_list = node_list2[:int(list_length2 / number_of_groups2)]
                elif i == number_of_groups2 - 1:
                    sub_node_list = node_list2[int(list_length2 / number_of_groups2) * i:]
                else:
                    sub_node_list = node_list2[int(list_length2 / number_of_groups2) *
                                    i:int(list_length2 / number_of_groups2) * ( i + 1)]
                sub_graph = G2.subgraph(sub_node_list).copy()
                graph_list.append(sub_graph)

            number_of_groups = number_of_groups + number_of_groups2

        matching_list = []
        # print("Start the multiprocessing with " + str(number_of_groups) + " tasks")
        #print("There are %d cpu in this machine" % multiprocessing.cpu_count())
        start_time = time.time()
        output_list = multiprocessing.Queue()
        jobs = []
        for i in range(number_of_groups):
            process = multiprocessing.Process(target=self.max_weight_matching,args=(graph_list[i],output_list,weight))
            jobs.append(process)
            process.start()

        for i in range(number_of_groups):
            matching_list += output_list.get()

        for j in jobs:
            j.join()

        end_time = time.time()
        # print("Total time for maximum weight matching is " + str(end_time - start_time))

        return matching_list

    def max_weight_matching(self,graph,output_list,weight_ = None):
        """
        Function for doing maximum weight matching
        :param graph: graph in networkx format
        :return: list of matching edges
        """
        #print("start processing " + str(os.getpid()))
        if output_list == None:
            if weight_ == None:
                return nx.max_weight_matching(graph)
            else:
                return nx.max_weight_matching(graph, weight=weight_)
        else:
            if weight_ == None:
                result = nx.max_weight_matching(graph)
            else:
                result = nx.max_weight_matching(graph, weight=weight_)
            output_list.put(result)

    def greedy_max_weight_matching(self, graph, weight):
        """
        Function to calculate max weight matching with greedy algorithm
        :param graph: graph in NetworkX format
        :param weight: weight for edges
        :return: list of matching edges
        """
        curr_matches = set()
        new_matching = []
        edge_weight_dict = nx.get_edge_attributes(graph, weight)
        edge_weight_list = [(edge_weight_dict[i], i) for i in edge_weight_dict]
        edge_weight_list.sort(reverse=True)
        for i in edge_weight_list:
            edge = i[1]
            if edge[0] in curr_matches or edge[1] in curr_matches:
                continue
            new_matching.append(edge)
            curr_matches.add(edge[0])
            curr_matches.add(edge[1])

        return new_matching

    def graph_random_split(self,split_factor, graph):
        """
        Function to split graph
        :param split_factor:
        :return:
        """
        G = graph
        node_list = list(G.nodes)
        number_nodes_subgraph = int(len(node_list) * split_factor)
        sub_node_list_1 = random.sample(node_list, k=number_nodes_subgraph)
        sub_node_list_2 = list(set(node_list) - set(sub_node_list_1))

        sub_graph_1 = G.subgraph(sub_node_list_1).copy()
        sub_graph_2 = G.subgraph(sub_node_list_2).copy()

        return sub_graph_1, sub_graph_2

    def segmentation_efficiency_loss(self, total_distance_saving, total_distance, weight=None):
        # Consider different market segmentation
        for i in range(10):
            start_time = time.time()
            split_factor = (i + 1) * 0.05
            graph1, graph2 = self.graph_random_split(split_factor)
            matching_results = self.heuristic_max_weight_matching(G1=graph1, G2=graph2, weight=weight)
            split_distance_saving = 0
            for edge in matching_results:
                distance_saving = self.graph[edge[0]][edge[1]][weight]
                split_distance_saving += distance_saving
            end_time = time.time()
            print("Total running time for this market segmentation is :" + str(end_time - start_time))
            print("The market segmentation factor is " + str(split_factor))
            print("The distance saving after market segmentation is: " + str(split_distance_saving))
            print("Efficiency loss is: " + str(total_distance_saving - split_distance_saving))
            print("Absolute efficiency loss rate is: " + str(
                (total_distance_saving - split_distance_saving) / total_distance))
            print("Relative efficiency loss rate is: " + str(
                (total_distance_saving - split_distance_saving) / total_distance_saving))
            print("------------------------------------------")

    def graph_sampling(self, scaling_factor):
        """
        Function to sample graph
        :param scaling_factor:
        :return: graph in the networkx format
        """
        G = self.graph
        node_list = list(G.nodes)
        number_nodes_subgraph = int(len(node_list) * scaling_factor)
        sub_node_list = random.sample(node_list, k=number_nodes_subgraph)

        total_distance = 0
        for i in sub_node_list:
            total_distance += int(self.trip_info[i][4])

        sub_graph = G.subgraph(sub_node_list).copy()

        return sub_graph, total_distance

    def graph_geo_split(self, G):
        """
        Function to split the graph geographically
        :return:
        """
        # self.trip_info
        # Key: trip_id, Value: [time,from,to,duration,length]

        # Read the geo split for road nodes
        raw_nodes_geo_split = pd.read_csv("data/points_geo_split.csv", header=None).values.tolist()
        nodes_geo_split = {}
        for i in raw_nodes_geo_split:
            nodes_geo_split[i[0]] = i[1]

        node_list = list(G.nodes)
        sub_node_list_1 = []
        for node in G.nodes:
            origin_node = int(self.trip_info[node][1])
            if nodes_geo_split[origin_node] == 'lower':
                sub_node_list_1.append(node)

        sub_node_list_2 = list(set(node_list) - set(sub_node_list_1))

        print("first cluster has " + str(len(sub_node_list_1)) + " nodes")
        print("second cluster has " + str(len(sub_node_list_2)) + " nodes")

        sub_graph_1 = G.subgraph(sub_node_list_1).copy()
        sub_graph_2 = G.subgraph(sub_node_list_2).copy()

        return sub_graph_1, sub_graph_2

    def visual_geo_split(self):
        nodes = {}  # Key: trip_id, Value: [time,from,to,duration,length]
        with open('data/points.csv') as g:
            wt = csv.reader(g, delimiter=',')
            for row in wt:
                nodes[int(row[0])] = (float(row[1]), float(row[2]))

        fig = plt.figure(figsize=(20, 10))
        for i in nodes:
            x, y, a, b = utm.from_latlon(nodes[i][0], nodes[i][1])
            if i <= 1470:
                plt.scatter(x, y, s=3, c='b', marker='o')
            else:
                plt.scatter(x, y, s=3, c='r', marker='x')
        plt.show()
        fig.savefig('geo_cluster.png')

    def generate_sub_graph(self, node_list):
        """
        Function to generate sub graph in networkx format given list of nodes
        :return: graph in networkx format
        """
        G = nx.Graph()
        for node in node_list:
            try:
                information = self.raw_data[node]
            except:
                continue
            index = int(len(self.raw_data[node]) / 4)
            node_distance = int(self.trip_info[node][4])
            node_time = int(self.trip_info[node][3])
            for i in range(index):
                node_b = node + int(information[i * 4])
                if node_b > node_list[-1]:
                    continue
                node_b_distance = int(self.trip_info[node_b][4])
                node_b_time = int(self.trip_info[node_b][3])
                travel_time_save = int(information[i * 4 + 2])
                travel_distance_save = int(information[i * 4 + 3])
                node_price = self.booking_fare + node_distance / 1609.34 * self.pool_pricing_distance \
                             + node_time / 60 * self.pool_pricing_time
                node_b_price = self.booking_fare + node_b_distance / 1609.34 * self.pool_pricing_distance \
                               + node_b_time / 60 * self.pool_pricing_time
                if node_price < self.min_price:
                    node_price = self.min_price
                if node_b_price < self.min_price:
                    node_b_price = self.min_price
                profit = node_price + node_b_price - self.booking_fare \
                         - (node_distance + node_b_distance - travel_distance_save) / 1609.34 * self.x_pricing_distance \
                         - (node_time + node_b_time - travel_time_save) / 60 * self.x_pricing_time
                G.add_edge(node, node_b, time=travel_time_save, distance=travel_distance_save, profit=profit)

        return G


if __name__ == '__main__':
    main()