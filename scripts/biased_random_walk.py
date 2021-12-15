"""
Taking the precomputed datasets, run a random-walk simulation where we start
from some initial system state and attempt to reach another state with a given
topology.

Allow the simulation to specify details about the loss function, which betti
curves to include, etc.

"""

import itertools

import numpy as np


def l2_loss(curve, target):
    return np.mean(np.square(target-curve))


def l1_loss(curve, target):
    return np.mean(np.abs(target-curve))


class RandomWalker:

    def __init__(self, data_files):
        sorted_data_files = sorted(data_files)
        var = np.load(sorted_data_files[0], allow_pickle=True)["var"][1:]
        self.chiN = var
        self.frac = []
        self.betti_curve_map = np.empty((len(sorted_data_files), len(var)), dtype=object)
        for idx, dataset in enumerate(sorted_data_files):
            frac = dataset.split("_frac-")[-1].split("_")[0]
            self.frac.append(frac)
            data = np.load(dataset, allow_pickle=True)
            for key in data.keys():
                if "betti" not in key:
                    continue
                for jdx in range(len(var)):
                    if type(self.betti_curve_map[idx, jdx]) != dict:
                        self.betti_curve_map[idx, jdx] = {}
                    self.betti_curve_map[idx, jdx][key] = data[key][jdx+1]

    def get_tuple_info(self, tuple):
        return (self.frac[tuple[0]], self.chiN[tuple[1]])

    def compute_loss(self, current_tuple, target_tuple, loss_func=l1_loss):
        current_data = self.betti_curve_map[current_tuple]
        target_data = self.betti_curve_map[target_tuple]

        loss = 0.0

        for key in current_data.keys():
            loss += loss_func(current_data[key], target_data[key])

        return loss

    def check_move_ok(self, current_tuple, move):
        shape = self.betti_curve_map.shape
        post_move = (current_tuple[0] + move[0], current_tuple[1] + move[1])
        for size, idx in zip(shape, post_move):
            if idx < 0 or idx >= size:
                return None
        return post_move

    def check_convex_op_possible(self, starting_tuple, target_tuple):
        moves = [-1, 0, 1]
        moves = list(itertools.product(moves, moves))
        moves.remove([0,0])

        current_tuple = starting_tuple
        loss = self.compute_loss(current_tuple, target_tuple)
        next_tuple = current_tuple

        while next_tuple is not None:

            next_tuple = None

            for move in moves:
                some_tuple = self.check_move_ok(current_tuple, move)
                if some_tuple is None:
                    continue
                new_loss = self.compute_loss(some_tuple, target_tuple)
                if new_loss < loss:
                    loss = new_loss
                    next_tuple = some_tuple
            
            if next_tuple == target_tuple:
                return True
        
        return False
        
            

