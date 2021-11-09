"""
Calculate cubic filtration in persistent homology from a mesh file
"""

import argparse
import pathlib

import gudhi
import gudhi.representations
import numpy as np

betti = gudhi.representations.vector_methods.BettiCurve(sample_range=[0, 1]).__call__

valid_input_suffixes = [".npz"]
valid_output_suffixes = [".npz"]

# argument parser
parser = argparse.ArgumentParser(description="Construct a cubic filtration from a density mesh.")
parser.add_argument('ifile', type=str, help=f'Input file (allowed formats: {valid_input_suffixes})')
parser.add_argument('ofile', type=str, help=f'Output file (allowed formats: {valid_output_suffixes})')

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

data = np.load(ifile)
mesh = data["mesh"]

out_data = {}

for i in range(0, len(mesh)):
    print(i)

    dimensions = mesh[i].shape

    a_conc = mesh[i].flatten()
    b_conc = 1 - a_conc

    cc_a = gudhi.CubicalComplex(dimensions=dimensions, top_dimensional_cells=a_conc)
    cc_b = gudhi.CubicalComplex(dimensions=dimensions, top_dimensional_cells=b_conc)
    cc_a.compute_persistence()
    cc_b.compute_persistence()

    for dim in range(len(dimensions)):
        bd = f"bd_a_{dim}"
        bet = f"betti_a_{dim}"
        bd_calc = cc_a.persistence_intervals_in_dimension(dim)
        if len(bd_calc) == 0:
            option_bd = None
            option_betti = None
        else:
            option_bd = bd_calc
            option_betti = betti(option_bd)
        if i == 0:
            out_data[bd] = [option_bd]
            out_data[bet] = [option_betti]
        else:
            out_data[bd].append(option_bd)
            out_data[bet].append(option_betti)

        bd = f"bd_b_{dim}"
        bet = f"betti_b_{dim}"
        bd_calc = cc_b.persistence_intervals_in_dimension(dim)
        if len(bd_calc) == 0:
            option_bd = None
            option_betti = None
        else:
            option_bd = bd_calc
            option_betti = betti(option_bd)
        if i == 0:
            out_data[bd] = [option_bd]
            out_data[bet] = [option_betti]
        else:
            out_data[bd].append(option_bd)
            out_data[bet].append(option_betti)

out_data["var"] = data["var"]

np.savez(ofile, **out_data)