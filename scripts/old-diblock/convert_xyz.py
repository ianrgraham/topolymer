import numpy as np
import os
import glob

rdir = "predict_l_n_2.0"
subdirs = ["0.16", "0.2", "0.28", "0.32", "0.36", "0.44"]

for sub in subdirs:
    ndir = os.path.join(rdir,sub)
    ssdirs = glob.glob(ndir+"/*")
    for ssub in ssdirs:
        print(ssub)
        files = glob.glob(ssub+"/ty*")
        files = [f for f in files if ".txt" not in f and ".npy" not in f]
        
        for old_f in files:
            print(old_f)
            data = np.loadtxt(old_f)

            new_f = old_f.replace("ty","xyz")
            N = len(data)
            with open(new_f,'w') as f:
                f.write(f"{N}\n")
                f.write("\n")
            with open(new_f,'a') as f:
                np.savetxt(f,data)
                
