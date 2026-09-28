# -*- coding: utf-8 -*-
"""
Data loading utilities for LoNALP.

The expected data layout under the --data_folder / --attribute_folder /
--alignment_folder directories is:

    graph/   <name>1.edges, <name>2.edges          (edge lists of G1, G2)
    attribute/<name>attr1.csv, <name>attr2.csv      (node attribute matrices)
    alignment/<name>.csv                            (ground-truth node alignment)

Node ids are assumed disjoint across the two networks: G2 nodes are stored
offset by (max(G1 node id) + 1) when building the union network.
"""
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


def loadG(data_folder, filename):
    """Load the two PPI networks G1 and G2 from edge-list files."""
    G1 = nx.Graph()
    G2 = nx.Graph()
    G1_edges = pd.read_csv(data_folder + filename + '1.edges', names=['0', '1'])
    G1.add_edges_from(np.array(G1_edges))
    G2_edges = pd.read_csv(data_folder + filename + '2.edges', names=['0', '1'])
    G2.add_edges_from(np.array(G2_edges))
    return G1, G2


def load_attribute(attribute_folder, filename, G1, G2):
    """
    Load node-attribute matrices and return their cosine-similarity matrix
    (attr_cos), together with the raw attribute matrices attr1, attr2.
    """
    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())
    attribute1 = pd.read_csv(attribute_folder + filename + 'attr1.csv', header=None, index_col=0)
    attribute2 = pd.read_csv(attribute_folder + filename + 'attr2.csv', header=None, index_col=0)
    attribute1 = np.array(attribute1.loc[G1_nodes, :])
    attribute2 = np.array(attribute2.loc[G2_nodes, :])
    attr_cos = cosine_similarity(attribute1, attribute2)
    return attr_cos, attribute1, attribute2
