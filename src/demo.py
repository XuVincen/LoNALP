# -*- coding: utf-8 -*-
from LoNALP import LoNALP
import pandas as pd
from utils import *
import warnings
import argparse
warnings.filterwarnings('ignore')


def read_alignment(alignment_folder, filename):
    alignment = pd.read_csv(alignment_folder + filename + '.csv', header=None)
    alignment_dict = {}
    alignment_dict_reversed = {}
    for i in range(len(alignment)):
        alignment_dict[alignment.iloc[i, 0]] = alignment.iloc[i, 1]
        alignment_dict_reversed[alignment.iloc[i, 1]] = alignment.iloc[i, 0]
    return alignment_dict, alignment_dict_reversed


def read_attribute(attribute_folder, filename, G1, G2):
    try:
        attribute, attr1, attr2 = load_attribute(attribute_folder, filename, G1, G2)
        attribute = attribute.transpose()
    except:
        attr1 = []
        attr2 = []
        attribute = []
        print('Attribute files not found.')
    return attribute, attr1, attr2


def parse_args():
    parser = argparse.ArgumentParser(description="Run LoNALP (KM alignment).")
    parser.add_argument('--attribute_folder', nargs='?', default='../attribute/')
    parser.add_argument('--data_folder', nargs='?', default='../graph/')
    parser.add_argument('--alignment_folder', nargs='?', default='../alignment/')
    parser.add_argument('--filename', nargs='?', default='bigtoy')
    parser.add_argument('--alpha', type=int, default=5)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--align_train_prop', type=float, default=0.0)
    parser.add_argument('--q', type=float, default=0.5)
    parser.add_argument('--c', type=float, default=0.5)
    parser.add_argument('--multi_walk', type=bool, default=False)
    return parser.parse_args()


def main(args):
    alignment_dict, alignment_dict_reversed = read_alignment(args.alignment_folder, args.filename)
    G1, G2 = loadG(args.data_folder, args.filename)
    attribute, attr1, attr2 = read_attribute(args.attribute_folder, args.filename, G1, G2)
    S, precision, seed_l1, seed_l2, best_acc = LoNALP(
        G1, G2, args.q, attr1, attr2, attribute, alignment_dict, alignment_dict_reversed,
        args.layer, args.align_train_prop, args.alpha, args.c, args.multi_walk)


if __name__ == '__main__':
    args = parse_args()
    main(args)
