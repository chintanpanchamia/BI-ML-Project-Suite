import sys
import collections
import time
import csv
import math
import copy
import numpy as np
import pandas as pd
import igraph as gp
import os

#from itertools import izip

#from sklearn.metrics.pairwise import cosine_similarity

import colour
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt



# Set b

b = 64

# Input parsing

if len(sys.argv) != 2:
    print "The input format is not proper ! Please enter in the following format."
    print "python anomaly.py <dataset directory containing one dataset>"
    exit(1)
data_dir = sys.argv[1]



# returns the hamming_distance

def hamming_distance(hash1, hash2):
    x = (hash1 ^ hash2) & ((1 << b) - 1)
    total = 0
    while x:
        total += 1
        #reset the last non zero bit to 0
        x &= x-1
    return total







def main():
    start_time = time.clock()
    generate_anomaly_series()
    print time.clock() - start_time, "seconds"



def series_plot(series, threshold_u, threshold_l):
    panda_series = pd.Series(series)
    plt.plot(panda_series)
    plt.axhline(y=threshold_u, color='r')
    plt.axhline(y=threshold_l, color='g')
    plt.savefig("series_graph.pdf")


def graph_plot(graph):
    layout = graph.layout("kk")
    gp.plot(graph, layout=layout)

def my_simhash(weighted_features):
    ans = [0] * b
    for t in [(my_hash(w[0]), w[1]) for w in weighted_features]:
        my_mask = 0
        for i in xrange(b):
            my_mask = 1 << i
            if t[0] & my_mask:
                ans[b - i - 1] += t[1]
            else:
                ans[b - i - 1] += -t[1]
    fp_binary = 0



    for i in xrange(b):
        if ans[i] >= 0:
            fp_binary += 1 << i
    return fp_binary



def my_hash(feature):

    if feature == "":

        return 0

    else:

        x = ord(feature[0])<<7

        #m = 10**9 + 7

        m = 100003

        my_mask = (1<<b) - 1

        for c in feature:

            x = ((x*m)^ord(c)) & my_mask

        x ^= len(feature)

        if x == -1:

            x = -2

        return x



def sim_function(u, v):

    return float(b - hamming_distance(u, v)) / b



def graph_similarity(graph1_weighted_features, graph2_weighted_features):

    u = my_simhash(graph1_weighted_features)

    v = my_simhash(graph2_weighted_features)

    return sim_function(u, v)



# Convert graphs to weighted featureslist of tuples

def convert_to_features(input_graph):

    weighted_features = list()

    #find the page ranks, tune the parameters later.

    page_rank_array = input_graph.pagerank(vertices=None, directed=True, weights=input_graph.es['weight'])

    #print page_rank_array

    # At this point, the page ranks are found for all the vertices, whether they are present or not. Check it!



    # Do the vertices, only the given vertices or all the rage! TODO

    for vertex in input_graph.vs:

        #print vertex.index

        weighted_features.append((str(vertex.index), page_rank_array[vertex.index]))



    # Do the edges: Sum the weights. Multiply with page rank.

    for edge in input_graph.es:

        u = edge.source

        v = edge.target

        edge_quality = float(page_rank_array[u])/(input_graph.vs[u].outdegree())

        weighted_features.append((str(u) + "_" + str(v), edge_quality))

    return weighted_features



def calc_threshold(result_series):

    count = 0

    for i in xrange(1, len(result_series)):

        count += abs(result_series[i] - result_series[i - 1])

    M = (float)(count)/(len(result_series) - 1)

    pd_series = pd.Series(result_series)

    median = pd_series.median()

    upper = median + 3 * M

    lower = median - 3 * M

    return upper, lower



def generate_anomaly_series():

    #main loop through all the graphs

    graphs_array = list()

    for filename in os.listdir(data_dir):

        #print data_dir + filename

        # Create the relative path properly TODO

        current_graph = gp.Graph.Read_Edgelist(data_dir + filename, directed=True)

        current_graph.es["weight"] = 1

        graphs_array.append(current_graph)



    #graph_plot(graphs_array[0])

    #print graphs_array[0].vs[1].neighbors(mode='OUT')

    #exit(0)



    if(len(graphs_array) == 1):

        print "Can't find series, only one graph !!! Abort."

        exit(0)



    result_series = list()



    graph_weighted_features = [convert_to_features(g) for g in graphs_array]



    for i in xrange(len(graphs_array) - 1):

        print i

        #graph1 = graphs_array[i]

        #graph2 = graphs_array[i + 1]

        #graph1_weighted_features = convert_to_features(graph1)

        #graph2_weighted_features = convert_to_features(graph2)

        graph_sim = graph_similarity(graph_weighted_features[i], graph_weighted_features[i+1])

        result_series.append(graph_sim)



    print "haha"

    #exit(0)



    print result_series



    [upper, lower] = calc_threshold(result_series)



    print upper, lower



    series_plot(result_series, upper, lower)



    # Write to file or plot whatever!







# Call the main. Entry point.



if __name__ == "__main__":

    main()