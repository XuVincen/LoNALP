# -*- coding: utf-8 -*-
"""
Weighted-node2vec embedding on the union PPI network, as described in the
LoNALP manuscript (Eq. (4)).

The union network merges G1 and G2 together with cross-network alignment
edges.  On bigtoy (no BLAST bit-scores) the alignment-edge weight omega is
provided by the structural (PP-distance) similarity, standing in for the
BLAST similarity used on real data.
"""
import numpy as np
import networkx as nx
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec


def cal_degree_dict(G, layer):
    """Multi-layer neighborhood degree profile for structural similarity."""
    G_degree = dict(G.degree())
    d = {0: {n: {n} for n in G.nodes()}}
    for i in range(1, layer + 1):
        d[i] = {}
        for node in G.nodes():
            ns = set()
            for nb in d[i - 1][node]:
                ns.update(G.neighbors(nb))
            for j in range(i - 1, -1, -1):
                ns -= d[j][node]
            d[i][node] = ns
    # convert to degree lists
    out = {}
    for i in range(layer + 1):
        out[i] = {}
        for node in G.nodes():
            out[i][node] = sorted([G_degree[x] for x in d[i][node]]) if d[i][node] else [0]
    return out


def struct_similarity(G1, G2, layer=3, alpha=5):
    """
    PP-distance structural similarity between G1 and G2 nodes.
    Returns a dense (n1, n2) similarity matrix plus the node lists.
    """
    G1_nodes = sorted(set(G1.nodes()))
    G2_nodes = sorted(set(G2.nodes()))
    n1, n2 = len(G1_nodes), len(G2_nodes)

    d1 = cal_degree_dict(G1, layer)
    d2 = cal_degree_dict(G2, layer)

    pp = np.zeros((n1, n2), dtype=float)
    for l in range(layer + 1):
        L1 = np.array([np.log(np.max(d1[l][x]) + np.e if d1[l][x] else np.e) for x in G1_nodes])
        L2 = np.array([np.log(np.max(d2[l][x]) + np.e if d2[l][x] else np.e) for x in G2_nodes])
        pp += np.abs(L1[:, None] - L2[None, :])
    for l in range(layer + 1):
        L1 = np.array([np.log(np.min(d1[l][x]) + 1 if d1[l][x] else 1) for x in G1_nodes])
        L2 = np.array([np.log(np.min(d2[l][x]) + 1 if d2[l][x] else 1) for x in G2_nodes])
        pp += np.abs(L1[:, None] - L2[None, :])
    pp /= 2
    sim = np.exp(-alpha * pp)
    return sim, G1_nodes, G2_nodes


def node2vec_walk(G, start, walk_length, p, q):
    """Single weighted node2vec walk (Eq. (4) semantics)."""
    walk = [start]
    while len(walk) < walk_length:
        cur = walk[-1]
        nbrs = list(G.neighbors(cur))
        if not nbrs:
            break
        if len(walk) == 1:
            weights = [G[cur][x].get('weight', 1.0) for x in nbrs]
        else:
            t = walk[-2]
            weights = []
            for x in nbrs:
                w = G[cur][x].get('weight', 1.0)
                if x == t:
                    coef = 1.0 / p
                elif G.has_edge(t, x):
                    coef = 1.0
                else:
                    coef = 1.0 / q
                weights.append(w * coef)
        weights = np.array(weights, dtype=float)
        s = weights.sum()
        probs = weights / s if s > 0 else np.full(len(nbrs), 1.0 / len(nbrs))
        walk.append(nbrs[int(np.random.choice(len(nbrs), p=probs))])
    return walk


def build_union_graph(G1, G2, mul, seed_list1, seed_list2, anchor_pairs):
    """
    Build the edge-weighted union network.

    Nodes are stringified; G2 nodes are offset by (mul+1).  Anchor/seed
    alignment edges carry weight = similarity (omega), intra-network edges
    carry weight 1.0.
    """
    U = nx.Graph()
    for u, v in G1.edges():
        U.add_edge(str(u), str(v), weight=1.0)
    for u, v in G2.edges():
        U.add_edge(str(u + mul + 1), str(v + mul + 1), weight=1.0)
    # alignment edges (anchor + confirmed seeds), weight = similarity
    for u, v, w in anchor_pairs:
        U.add_edge(str(u), str(v + mul + 1), weight=float(w))
    for u, v in zip(seed_list1, seed_list2):
        if U.has_edge(str(u), str(v + mul + 1)):
            U[str(u)][str(v + mul + 1)]['weight'] = 1.0
        else:
            U.add_edge(str(u), str(v + mul + 1), weight=1.0)
    return U


def lonalp_embed(G1, G2, mul, seed_list1, seed_list2,
                 num_anchors=20, p=10, q=2, layer=3, alpha=5,
                 num_walks=20, walk_length=80, window=5, vector_size=64):
    """
    End-to-end LoNALP embedding:
      structural similarity -> anchor pairs -> union graph -> weighted node2vec
    Returns the gensim Word2Vec model.
    """
    sim, G1_nodes, G2_nodes = struct_similarity(G1, G2, layer, alpha)
    # anchor pairs = top-num_anchors highest-similarity node pairs
    flat = [(int(i), int(j), sim[i, j]) for i in range(sim.shape[0]) for j in range(sim.shape[1])]
    flat.sort(key=lambda x: -x[2])
    anchor_pairs = [(G1_nodes[i], G2_nodes[j], w) for i, j, w in flat[:num_anchors]]

    U = build_union_graph(G1, G2, mul, seed_list1, seed_list2, anchor_pairs)

    nodes = list(U.nodes())
    walks = []
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for n in nodes:
            walks.append(node2vec_walk(U, n, walk_length, p, q))

    model = Word2Vec(walks, vector_size=vector_size, window=window,
                     min_count=0, hs=1, sg=1, negative=10, workers=4, epochs=5)
    return model
