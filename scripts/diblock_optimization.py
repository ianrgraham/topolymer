#! env python

from cmath import inf
import cmaes
import lammps_init_v2 as lmp
import numpy as np
from itertools import product
import os
import shutil
import subprocess
import math

def last_frame(input, output):
    command = "sed -n '/^ITEM: TIMESTEP$/{h;b};H;${x;p}'" + f" {input} > {output}"
    os.system(command)


def myround(x, prec=2, base=.05):
  return round(base * round(float(x)/base),prec)

DIM = 3
L = 16.0

REF_SNAPSHOT = "asjdflasjlfjasdf"

resolution = 1
boxes = [ np.array([L, L, L])  ]

desired_rho0 = 5
N = np.array([20]) # Total chain length; N_a = f_a * N

SMEAR = [3, 1.0]

optimization_generations = 50

def compute_cubic_phom(lammpstrj_snapshot):
    mesh_output = "mesh.npz"
    command = f"python ../scripts/mesh_lammps.py --input {lammpstrj_snapshot} --output {mesh_output} " \
        f"--dim {DIM} --bins {int(L)} " \
        f"--smear {SMEAR}"

    subprocess.run(command.split())

    phom_output = "phom.npz"

    command = f"python ../scripts/cubic_phom.py {mesh_output} {phom_output}"

    data = np.load(phom_output, allow_pickle=True) # phom data for this snapshot
    return data

ref_phom = compute_cubic_phom(REF_SNAPSHOT)

# TODO function to compute loss from topology
def topology_loss(lammpstrj_snapshot, ref):

    data = compute_cubic_phom(lammpstrj_snapshot)

    # betti_a_0
    # betti_b_0
    # betti_a_1
    # betti_b_1
    # betti_a_2
    # betti_b_2

    loss = 0.0

    for key in data:
        ref_betti = ref[key]
        data_betti = data[key]
        loss += np.sum(np.square(data_betti - ref_betti))


    return loss

dir_path = os.path.dirname(os.path.realpath(__file__))
count = 0

for n, box_dim in zip(N, boxes):
    print(f"n {n}")

    init_chiN = 5
    init_x = np.array([0.1, init_chiN])
    optimizer = cmaes.CMA(
        mean=init_x,
        sigma=1.0,
        bounds=np.array([[0.025, 0.5], [0.0, np.inf]]),
        population_size=4)  # build CMA optimizer

    # loop over generations
    for generation in range(optimization_generations):

        # calculate solutions for the current generation
        solutions = []
        for _ in range(optimizer.population_size):
            os.chdir(dir_path)
            x = optimizer.ask()  # ask for parameters

            x[0] = myround(x[0],prec=2, base=0.05)
            frac = x[0]
            chiN = x[1]

            print(x)
            # if math.isclose(frac,0.0) or math.isclose(frac,1.0):
            #     solutions.append((x,inf))
            #     continue


            box = lmp.input_config(box_dim[0],box_dim[1],box_dim[2])
            box.periodic = False

            box.add_diblock_rho0(1, 2, frac, n, desired_rho0, 1, 1)

            Nx = int(box_dim[0] // resolution)
            Ny = int(box_dim[1] // resolution)
            Nz = int(box_dim[2] // resolution)

            config_file = f"conf_files/Nb_{n}" + (3*"_{}").format(box_dim[0],box_dim[1],box_dim[2]) \
                        + f"_diblock_f_{frac}_rho_{desired_rho0}.data"
            
            box.write(config_file) # Generates the initial configuration


            # TODO build the command that will run the simulation
            epoch_folder = f"epochs/{count}"
            os.makedirs(epoch_folder, exist_ok=True)
            input_file = "sample_input.gputild"
            shutil.copy2(input_file, epoch_folder)

            input_path = os.path.join(dir_path,epoch_folder,input_file)

            kappa = 5
            chi = chiN/n
            kappa_chi = 2*kappa + chi

            CONF_FOLDER="{CONF_DIR}"
            INPUT_FILE="{INPUT_FILE}"
            nx="{NX}"
            ny="{NY}"
            nz="{NZ}"
            self_int="{POLY_SELF_AMP}"
            cross_int="{POLY_CROSS_AMP}"
            conf_folder = os.path.join(dir_path, "conf_files")
            command=f"sed -i -e 's|{CONF_FOLDER}|{conf_folder}|g' -e 's|{INPUT_FILE}|{config_file}|'\
                -e 's|{nx}|{Nx}|' -e 's|{ny}|{Ny}|' -e 's|{nz}|{Nz}|'\
                    -e 's|{self_int}|{kappa}|' -e 's|{cross_int}|{kappa_chi}|'  {input_path}"
            # print(command)
            subprocess.call([command], shell=True)
            os.chdir(epoch_folder)
            command = f"echo 'gpu-tild -in {input_file}'"
            return_val = subprocess.call(command.split())
            print(return_val)

            lammpstrj_file = "traj.lammpstrj"
            output_lammpstrj_snapshot = "snapshot.lammpstrj"

            last_frame(lammpstrj_file, output_lammpstrj_snapshot)

            loss = topology_loss(output_lammpstrj_snapshot, ref_phom)  # TODO compute some loss based on topology
            count = count+1
            solutions.append((x, loss))  # append params & loss to solution list
            # print(solutions)
        optimizer.tell(solutions)  # update the optimizer with solutions


