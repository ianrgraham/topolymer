import numpy as np
import os
import glob

rdir = ["N_30_sph"]
rdir.extend(glob.glob("early_frames/*"))

for ndir in rdir:
    ssdirs = glob.glob(ndir+"/*")
    for ssub in ssdirs:
        print(ssub)
        if os.path.exists(ssub+"/done"):
            print("it is done")
            #continue
        old_f = ssub + "/traj.lammpstrj"
	
        with open(old_f,'r') as of:
            ctime = 0
            cnum = 1000
            cbox = 10
            stream = False
            box = False
            time = False
            num = False
            for in_line in of.readlines():
                
                if "ITEM" in in_line:
                    if "TIMESTEP" in in_line:
                        time = True
                        if stream:
                            nf.close()
                            stream = False
                    elif "NUMBER OF" in in_line:
                        num = True
                    elif "BOX BOUNDS" in in_line:
                        box = True
                    elif "ATOMS id" in in_line:
                        stream = True
                elif time:
                    time = False
                    ctime = in_line.strip("\n")
                    print(ctime)
                    nf = open(ssub+f"/ty_{ctime}","w")
                elif num:
                    num = False     
                    cnum = in_line
                    #nf.write(f"{cnum}")
                    #nf.write("\n")
                elif box:
                    box = False
                    cbox = float(in_line.strip("\n").split(" ")[-1])
                elif stream:
                    split = in_line.split(" ")
                    #join = " ".join(split[3:])
                    join = " ".join([str(float(s)/cbox) for s in split[3:]])
                    nf.write(f"{split[1]} {join}\n")
                
        open(ssub+"/done", 'a').close()

