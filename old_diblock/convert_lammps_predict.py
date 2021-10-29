import gudhi
import numpy as np
import gudhi.representations
import os
import glob

rdir = ["N_30_sph"]
rdir.extend(glob.glob("early_frames/*"))

betti = gudhi.representations.vector_methods.BettiCurve(sample_range=[0, 1]).__call__

for ndir in rdir:
    ssdirs = glob.glob(ndir+"/*")
    for ssub in ssdirs:
        files = glob.glob(ssub+"/ty_*")
        files = [f for f in files if ".txt" not in f]
        for old_f in files:
            # convert to density format
            f = old_f + ".txt"
            os.system(f"./read_and_mesh/dmft 1.0 25 {old_f} {f}")
            
            # do same old processing
            data = np.loadtxt(f)
            a_conc = data[:,0]
            b_conc = data[:,1]
            cc_a = gudhi.CubicalComplex(dimensions=[25,25,25], top_dimensional_cells=a_conc)
            cc_b = gudhi.CubicalComplex(dimensions=[25,25,25], top_dimensional_cells=b_conc)
            cc_a.compute_persistence()
            cc_b.compute_persistence()
            for dim in [0,1,2]:
                bd_a = cc_a.persistence_intervals_in_dimension(dim)
                np.save(f.replace(".txt",f"_bd{dim}_a"), bd_a)
                np.save(f.replace(".txt",f"_bet{dim}_a"), betti(bd_a))

                bd_b = cc_b.persistence_intervals_in_dimension(dim)
                np.save(f.replace(".txt",f"_bd{dim}_b"), bd_b)
                np.save(f.replace(".txt",f"_bet{dim}_b"), betti(bd_b))
                
