#! env python

import cmaes
import lammps_init_v2 as lmp
import numpy as np
from itertools import product


Dim = 3

boxes = [ np.array([32.00, 32.00, 32.00])  ]

desired_rho0 = 5
N = np.array([20]) # Total chain length; N_a = f_a * N

fracs = np.array([0.5])

optimization_generations = 50

# TODO function to compute loss from topology

# TODO function to 

for n, frac in product(N, fracs):
    print(f"n {n}\tfrac {frac}")

    optimizer = CMA(mean=np.zeros(2), sigma=1.3)  # build CMA optimizer

    # loop over generations
    for generation in range(optimization_generations):

        # calculate solutions for the current generation
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()  # ask for parameters

            box = lmp.input_config(box_dim[0],box_dim[1],box_dim[2])
            box.periodic = False

            box.add_diblock_rho0(1, 2, frac, N, desired_rho0, 1, 1):

            config_file = f"conf_files/Nb_{nb}" + (3*"_{}").format(box_dim[0],box_dim[1],box_dim[2])
                    +f"_diblock_f_{frac}_rho_{desired_rho0}.data"
            
            box.write(config_file) # Generates the initial configuration

            command = f"./gpu-tild -in {input_file} -config {config_file}"

            # TODO build the command that will run the simulation

            loss = 1.0  # TODO compute some loss based on topology

            solutions.append((x, loss))  # append params & loss to solution list

        optimizer.tell(solutions)  # update the optimizer with solutions


