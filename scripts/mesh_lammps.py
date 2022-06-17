"""
Calculate and store grid densities from a lammps dump file
"""

import argparse
import itertools
import pathlib
import re
from typing import List, Optional, Tuple
import sys

import numpy as np
from numba import njit

@njit(fastmath=True)
def _gaus_smear(v1, v2, sigma):
    r = np.linalg.norm(v2-v1)
    return np.exp(-(r/sigma)**2)

@njit
def _wrap(vector, box, dim):
    out = np.zeros(dim)
    for i in range(dim):
        l2 = (box[i][1] - box[i][0])/2
        if vector[i] >= box[i][1]:
            out[i] = vector[i] - l2
        elif vector[i] < box[i][0]:
            out[i] = vector[i] + l2
        else:
            out[i] = vector[i]
    return out

@njit
def _add_smeared_gauss_to_grid(array_ref, idx_ref, disp, disp_x, A_count, B_count, smear_sigma, dim, box, bins, type):
    new_idx = np.zeros(dim, dtype=np.int64)
    for i in range(len(disp)):
        d = disp[i]
        dx = disp_x[i]
        coord = _wrap(array_ref + dx, box, dim)
        for i in range(dim):
            new_idx[i] = (idx_ref[i] + d[i])%bins[i]
        if type == 1:
            A_count[new_idx[0], new_idx[1]] += _gaus_smear(array_ref, coord, smear_sigma)
        elif type == 2:
            B_count[new_idx[0], new_idx[1]] += _gaus_smear(array_ref, coord, smear_sigma)
    

class MeshFramesGenerator:

    def __init__(self, bins: List[int], var: Optional[List] = None, smear: Optional[Tuple[int, float]] = None):
        self.bins = np.array(bins)
        self.dim = len(bins)
        if var is not None:
            assert len(var) == 2
        self.var = var
        self.state = None
        self.state_counter = None
        self.item_regex = re.compile("ITEM:(.*)$")
        self.atoms = None
        self.mesh = []
        self.mesh_grid = []
        self.grid = None
        self.dx = None
        self.box = None
        self.time = []
        self.A_count = None
        self.B_count = None

        if smear is not None:
            assert len(smear) == 2
        
        self.smear = smear

    def wrap(self, vector):

        assert len(vector) == self.dim
        out = _wrap(np.array(vector), self.box, self.dim)
        return out
        

    def ingest_line(self, line: str):
        search = self.item_regex.search(line)

        # overly complicated state machine to parse lammpstrj
        if search is not None:
            item = search.group(1)
            if "TIMESTEP" in item:
                assert self.state is None
                self.state = "time"
                self.state_counter = 1
            elif "NUMBER OF ATOMS" in item:
                assert self.state == "time"
                self.state = "num"
                self.state_counter = 1
            elif "BOX" in item:
                assert self.state == "num"
                self.state = "box"
                self.state_counter = 3
                self.grid = []
                self.dx = []
                self.box = []
            elif "ATOMS" in item:
                assert self.state == "box"
                assert len(self.box) == self.dim
                self.box = np.array(self.box)
                self.dx = np.array(self.dx)
                self.grid = np.array(self.grid)
                self.A_count = np.zeros(self.bins)
                self.B_count = np.zeros(self.bins)
                self.state = "atoms"
                self.state_counter = self.atoms - 1
                if self.smear is not None:
                    bins, _ = self.smear
                    disp = list(itertools.product(range(-bins, bins + 1), repeat=self.dim))
                    disp_x = []
                    for d in disp:
                        disp_x.append(d*self.dx)
                    self.disp = np.array(disp)
                    self.disp_x = np.array(disp_x)

            else:
                raise ValueError(f"Unknown item tag ({item}) found")
        else:
            if self.state == "time":
                print(int(line))
                self.time.append(int(line))
            elif self.state == "num":
                self.atoms = int(line)
            elif self.state == "box":
                axis_bounds = [float(b) for b in line.split()]
                if axis_bounds[0] != axis_bounds[1] and self.state_counter > 3 - self.dim:
                    self.box.append(axis_bounds)
                    bin_edges = np.linspace(axis_bounds[0], axis_bounds[1], self.bins[3-self.state_counter]+1)
                    self.grid.append(
                        bin_edges
                    )
                    self.dx.append(bin_edges[1] - bin_edges[0])
                
            elif self.state == "atoms":
                data = line.split()
                type = int(data[1])

                ref_coord = np.array([float(x) for x in data[3:3+self.dim]])
                coord = self.wrap(ref_coord)

                idx = np.array([np.digitize(coord[i], self.grid[i]) - 1 for i in range(self.dim)])

                if self.smear is None:
                    
                    if type == 1:
                        self.A_count[idx[0], idx[1]] += 1
                    elif type == 2:
                        self.B_count[idx[0], idx[1]] += 1
                    else:
                        raise ValueError(f"Found particle type other than 0 and 1: {type}")
                else:
                    _add_smeared_gauss_to_grid(coord, idx, self.disp, self.disp_x, self.A_count, self.B_count, self.smear[1], self.dim, self.box, self.bins, type)

                
            elif self.state is None:
                return
            else:
                raise Exception("Invalid internal state obtained")
            self.state_counter -= 1
            if self.state_counter < 0:
                if self.state == "atoms":
                    self.mesh.append(self.A_count/(self.A_count + self.B_count))
                    self.mesh_grid.append(self.grid)
                    self.state = None
                else:
                    raise Exception("Invalid internal state obtained")

    def dump(self, outfile: pathlib.Path):
        assert self.state is None
        if self.var is not None:
            var = np.linspace(self.var[0], self.var[1], len(self.mesh))
        else:
            var = None
        out_data = {"mesh": np.array(self.mesh), 
                    "mesh_grid": np.array(self.mesh_grid),
                    "time": np.array(self.time),
                    "var": var}
        np.savez(outfile, **out_data)


valid_input_suffixes = [".lammpstrj", ".tec"]
valid_output_suffixes = [".npz"]

# argument parser
parser = argparse.ArgumentParser(description="Construct a density mesh from a lammps file.")
parser.add_argument('--input', type=str, nargs="+", help=f'Input file(s) (allowed formats: {valid_input_suffixes})')
parser.add_argument('--output', type=str, help=f'Output file (allowed formats: {valid_output_suffixes})')
parser.add_argument('--dim', type=int, help="Dimensions of the system", default=2)
parser.add_argument('--bins', type=int, nargs='+', help='Bins of the mesh', default=[20])
parser.add_argument('--lin-var', type=int, nargs=2, help='Quantity to vary linearly that describes the state. '
                    'Specify as pair of starting and stopping values (start, stop).')
parser.add_argument('--smear', type=float, nargs=2, help="Tuple of arguements")
args = parser.parse_args()

ifiles = args.input
ofile = pathlib.Path(args.output)

if ofile.suffix not in valid_output_suffixes:
    raise ValueError(
        f"Output file type '{ofile.suffix}' is not supported\n\
            Supported output types: {valid_output_suffixes}"
    )

if len(ifiles) == 1:  # lammpstrj

    ifile = pathlib.Path(ifiles[0])

    if ifile.suffix not in valid_input_suffixes[:1]:
        raise ValueError(
            f"Input file type '{ifile.suffix}' is not supported\n\
                Supported input types: {valid_input_suffixes}"
        )

    bins = args.bins
    dim = args.dim

    var = args.lin_var
    smear = args.smear
    if smear is not None:
        smear = (int(smear[0]), float(smear[1]))

    assert dim == 2 or dim == 3, "Improper 'dim' given"

    assert var is None or len(var) == 2, "Improper arguments given to --lin-var"

    if len(bins) == 1:
        bins = bins*dim

    f = open(ifile,"r")
    lines = f.readlines()

    meshgen = MeshFramesGenerator(bins, var=var, smear=smear)

    for line in lines:
        # print(line)
        line = line.replace("\n","")
        meshgen.ingest_line(line)
        
    meshgen.dump(ofile)

else:  # tec files

    for ifile in ifiles:
        if ifile.suffix not in valid_input_suffixes[1:]:
            raise ValueError(
                f"Input file type '{ifile.suffix}' is not supported\n\
                    Supported input types: {valid_input_suffixes}"
            )