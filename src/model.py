# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def structing(layers, G1, G2, G1_degree_dict, G2_degree_dict, attribute, alpha, c):
    """
    Compute a structural-similarity matrix (PP-distance) between G1 and G2 nodes.

    Pure-numpy implementation (avoids pandas DataFrame alignment issues that
    arise after multi-round graph updates).
    """
    G1_nodes = sorted(set(G1.nodes()))
    G2_nodes = sorted(set(G2.nodes()))
    n1 = len(G1_nodes)
    n2 = len(G2_nodes)

    pp_dist = np.zeros((n1, n2), dtype=float)

    for layer in range(layers + 1):
        L1 = np.array([np.log(np.max(G1_degree_dict[layer][x]) + np.e) for x in G1_nodes])
        L2 = np.array([np.log(np.max(G2_degree_dict[layer][x]) + np.e) for x in G2_nodes])
        pp_dist += np.abs(L1[:, None] - L2[None, :])

    for layer in range(layers + 1):
        L1 = np.array([np.log(np.min(G1_degree_dict[layer][x]) + 1) for x in G1_nodes])
        L2 = np.array([np.log(np.min(G2_degree_dict[layer][x]) + 1) for x in G2_nodes])
        pp_dist += np.abs(L1[:, None] - L2[None, :])

    pp_dist /= 2
    pp_dist = np.exp(-alpha * pp_dist)

    if attribute is not None and len(attribute) and not np.allclose(np.array(attribute), np.array(attribute).flat[0]):
        # Only fuse attribute when it carries real signal (skip the all-1.0 toy attribute)
        attr = np.array(attribute, dtype=float)
        if attr.shape != (n1, n2):
            arr = np.zeros((n1, n2), dtype=float)
            h = min(n1, attr.shape[0])
            w = min(n2, attr.shape[1])
            arr[:h, :w] = attr[:h, :w]
            attr = arr
        pp_dist = c * pp_dist + attr * (1 - c)

    # top-10 structural neighbors per node
    struc_neighbor1 = {}
    struc_neighbor2 = {}
    struc_neighbor_sim1 = {}
    struc_neighbor_sim2 = {}

    for i in range(n1):
        row = pp_dist[i]
        idx = np.argsort(-row)[:10]
        struc_neighbor1[G1_nodes[i]] = [G2_nodes[j] for j in idx]
        w = row[idx]
        w = w / w.sum()
        struc_neighbor_sim1[G1_nodes[i]] = w

    pp_dist_t = pp_dist.T
    for i in range(n2):
        row = pp_dist_t[i]
        idx = np.argsort(-row)[:10]
        struc_neighbor2[G2_nodes[i]] = [G1_nodes[j] for j in idx]
        w = row[idx]
        w = w / w.sum()
        struc_neighbor_sim2[G2_nodes[i]] = w

    return struc_neighbor1, struc_neighbor2, struc_neighbor_sim1, struc_neighbor_sim2
