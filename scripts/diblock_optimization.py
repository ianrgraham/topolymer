#! env python

import cmaes
import lammps_init_v2 as lmp
import numpy as np
from itertools import product


Dim = 3

boxes = [ np.array([32.00, 32.00, 32.00])  ]

desired_rho0 = 5
N = np.array([20]) # Total chain length; N_a = f_a * N

optimization_generations = 50

# TODO function to compute loss from topology
def topology_loss(frac, chiN):
    return (frac - 0.5) ** 2 + (chiN - 20) ** 2

# TODO function to 

for n, box_dim in zip(N, boxes):
    print(f"n {n}")

    init_chiN = 5
    init_x = np.array([0.1, init_chiN])

    optimizer = cmaes.CMA(
        mean=init_x,
        sigma=0.2,
        bounds=np.array([[0.0, 0.5], [0.0, np.inf]]),
        population_size=10)  # build CMA optimizer

    # loop over generations
    for generation in range(optimization_generations):

        # calculate solutions for the current generation
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()  # ask for parameters

            frac = x[0]
            chiN = x[1]

            box = lmp.input_config(box_dim[0],box_dim[1],box_dim[2])
            box.periodic = False

            box.add_diblock_rho0(1, 2, frac, n, desired_rho0, 1, 1)

            config_file = f"conf_files/Nb_{n}" + (3*"_{}").format(box_dim[0],box_dim[1],box_dim[2]) \
                        + f"_diblock_f_{frac}_rho_{desired_rho0}.data"
            
            box.write(config_file) # Generates the initial configuration

            command = f"./gpu-tild -in {input_file} -config {config_file}"

            # TODO build the command that will run the simulation

            loss = topology_loss(frac, chiN)  # TODO compute some loss based on topology

            solutions.append((x, loss))  # append params & loss to solution list
        
        optimizer.tell(solutions)  # update the optimizer with solutions


