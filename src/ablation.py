# -*- coding: utf-8 -*-
"""
Ablation study for LoNALP — isolates the contribution of each component:
  (A) KM alignment          (use_km True/False)
  (B) weighted node2vec     (p=10,q=2  vs  p=1,q=1  unweighted)
  (C) feedback loop         (use_feedback True/False)

Runs each config over several random seeds and reports the peak alignment
accuracy (max 'All seed accuracy' over iterations) as the summary metric.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from utils import loadG, load_attribute
from LoNALP import LoNALP


def run_config(G1, G2, attr, a1, a2, ad, adr, seed, use_km, use_weighted, use_feedback):
    np.random.seed(seed)
    if use_weighted:
        p, q2 = 10, 2
    else:
        p, q2 = 1, 1
    S, precision, seed_l1, seed_l2, best_acc = LoNALP(
        G1, G2, 0.5, a1, a2, attr, ad, adr,
        layer=3, align_train_prop=0.0, alpha=5, c=0.5, multi_walk=False,
        use_km=use_km, use_feedback=use_feedback, p=p, q2=q2,
        max_iters=8, n_walks=15, walk_len=60)
    return best_acc


def main():
    from demo import read_alignment, read_attribute as ra
    data_folder = '../graph/'
    attribute_folder = '../attribute/'
    alignment_folder = '../alignment/'
    filename = 'bigtoy'

    ad, adr = read_alignment(alignment_folder, filename)
    G1, G2 = loadG(data_folder, filename)
    attr, a1, a2 = ra(attribute_folder, filename, G1, G2)
    attr = attr.transpose()

    configs = [
        ('LoNALP full (KM + weighted n2v + feedback)', dict(use_km=True, use_weighted=True, use_feedback=True)),
        ('  - KM (greedy)',                            dict(use_km=False, use_weighted=True, use_feedback=True)),
        ('  - weighted n2v (p=q=1)',                   dict(use_km=True, use_weighted=False, use_feedback=True)),
        ('  - feedback (single round)',                dict(use_km=True, use_weighted=True, use_feedback=False)),
    ]

    n_seeds = 3
    print('=== LoNALP ablation on bigtoy ({} seeds, avg alignment accuracy) ==='.format(n_seeds))
    results = {}
    for name, cfg in configs:
        accs = []
        for s in range(n_seeds):
            acc = run_config(G1.copy(), G2.copy(), attr, a1, a2, ad, adr, s,
                             cfg['use_km'], cfg['use_weighted'], cfg['use_feedback'])
            accs.append(acc)
        avg = np.mean(accs)
        results[name] = avg
        print('  {:45s} : {:.4f}  (per-seed: {})'.format(name, avg, [round(a, 4) for a in accs]))

    # report deltas
    full = results['LoNALP full (KM + weighted n2v + feedback)']
    print('\n=== Component contributions ===')
    print('  KM contribution            : {:.4f}'.format(full - results['  - KM (greedy)']))
    print('  weighted-n2v contribution  : {:.4f}'.format(full - results['  - weighted n2v (p=q=1)']))
    print('  feedback contribution      : {:.4f}'.format(full - results['  - feedback (single round)']))


if __name__ == '__main__':
    main()
