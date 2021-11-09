"""
Calculate and store grid densities from a lammps dump file
"""

import argparse
import pathlib
import re
from typing import List, Optional
import sys

import numpy as np

class MeshFramesGenerator:

    def __init__(self, bins: List[int], var: Optional[List]):
        self.bins = bins
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
        self.box = None
        self.time = []
        self.A_count = None
        self.B_count = None

    def wrap(self, vector):

        assert len(vector) == self.dim
        out = []
        for i in range(self.dim):
            l2 = (self.box[i][1] - self.box[i][0])/2
            if vector[i] >= self.box[i][1]:
                out.append(vector[i] - l2)
            elif vector[i] < self.box[i][0]:
                out.append(vector[i] + l2)
            else:
                out.append(vector[i])
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
                self.box = []
            elif "ATOMS" in item:
                assert self.state == "box"
                assert len(self.box) == self.dim
                self.A_count = np.zeros(self.bins)
                self.B_count = np.zeros(self.bins)
                self.state = "atoms"
                self.state_counter = self.atoms - 1
            else:
                raise ValueError(f"Unknown item tag ({item}) found")
        else:
            if self.state == "time":
                self.time.append(int(line))
            elif self.state == "num":
                self.atoms = int(line)
            elif self.state == "box":
                axis_bounds = [float(b) for b in line.split()]
                if axis_bounds[0] != axis_bounds[1] and self.state_counter > 3 - dim:
                    self.box.append(axis_bounds)
                    self.grid.append(
                        np.linspace(axis_bounds[0], axis_bounds[1], bins[3-self.state_counter]+1)
                    )
            elif self.state == "atoms":
                data = line.split()
                type = int(data[1])
                coord = [float(x) for x in data[3:3+self.dim]]
                coord = self.wrap(coord)
                idx = tuple([np.digitize(coord[i], self.grid[i]) - 1 for i in range(dim)])

                if type == 1:
                    self.A_count[idx] += 1
                elif type == 2:
                    self.B_count[idx] += 1
                else:
                    raise ValueError(f"Found particle type other than 0 and 1: {type}")
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


valid_input_suffixes = [".lammpstrj"]
valid_output_suffixes = [".npz"]

# argument parser
parser = argparse.ArgumentParser(description="Construct a density mesh from a lammps file.")
parser.add_argument('ifile', type=str, help=f'Input file (allowed formats: {valid_input_suffixes})')
parser.add_argument('ofile', type=str, help=f'Output file (allowed formats: {valid_output_suffixes})')
parser.add_argument('dim', type=int, help="Dimensions of the system")
parser.add_argument('--bins', type=int, nargs='+', help='Bins of the mesh', default=[20])
parser.add_argument('--lin-var', type=int, nargs=2, help='Quantity to vary linearly that describes the state. '
                    'Specify as pair of starting and stopping values (start, stop).')
args = parser.parse_args()

ifile = pathlib.Path(args.ifile)
ofile = pathlib.Path(args.ofile)

if ifile.suffix not in valid_input_suffixes:
    raise ValueError(
        f"Input file type '{ifile.suffix}' is not supported\n\
            Supported input types: {valid_input_suffixes}"
    )

if ofile.suffix not in valid_output_suffixes:
    raise ValueError(
        f"Output file type '{ofile.suffix}' is not supported\n\
            Supported output types: {valid_output_suffixes}"
    )

bins = args.bins
dim = args.dim

var = args.lin_var

assert dim == 2 or dim == 3, "Improper 'dim' given"

assert var is None or len(var) == 2, "Improper arguments given to --lin-var"

if len(bins) == 1:
    bins = bins*dim

f = open(ifile,"r")
lines = f.readlines()

meshgen = MeshFramesGenerator(bins, var=var)

for line in lines:
    # print(line)
    line = line.replace("\n","")
    meshgen.ingest_line(line)
    
meshgen.dump(ofile)