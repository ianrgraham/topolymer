import numpy as np
import pbc_utils as pbc
import math
import os

class input_config: 
    def __init__(self, xbox, ybox, zbox):
        self.natoms = 0
        self.nbonds = 0
        self.nmasses = 0
        self.ndihedrals = 0
        self.nimpropers = 0
        self.masses = []
        self.ang_types = []
        self.bond_types = []
        self.bonds = np.array([], dtype=np.int64).reshape(0,4)
        self.nbond_types = 0
        self.nangles = 0
        self.nang_types = 0
        self.x = np.array([], dtype=np.int64).reshape(0,6)
        self.RESID = np.zeros((0, 3), 'd')
        self.L = np.zeros(3, 'd')
        self.L[0] = float(xbox)
        self.L[1] = float(ybox)
        self.L[2] = float(zbox)
        self.lo = -(self.L)/2
        self.hi = (self.L)/2
        self.xlo = self.lo[0]
        self.ylo = self.lo[1]
        self.zlo = self.lo[2]
        self.xhi = self.hi[0]
        self.yhi = self.hi[1]
        self.zhi = self.hi[2]
        self.np_list = np.array([], dtype=np.int64).reshape(0,4)
        self.num_chns = 0
        self.periodic = False
        

    def __add_particle_type(self, part_type):
        if ( not part_type in  self.masses and not part_type == None ):
            self.masses.append(part_type)
            self.nmasses += 1

    def __add_bond_type(self, bond_type):
        if ( not bond_type in self.bond_types and not bond_type == None):
            self.bond_types.append(bond_type)
            self.nbond_types+= 1

    def __update_particle_count(self, count_new_atoms):
        self.natoms += count_new_atoms

    def __update_chain_count(self, count_new_chains):
        self.num_chns += count_new_chains 

    def __add_bond_check_bond_overlap(self,loc_array, index, monomer_increment, Lbond, rmin, old_index = None, rad = None):
        if old_index == None:
            old_index = index - 1
        theta = 2 * np.pi * np.random.random_sample()
        phi = np.pi * np.random.random_sample()

        dx = Lbond * np.sin(phi) * np.cos(theta)
        dy = Lbond * np.sin(phi) * np.sin(theta)
        dz = Lbond * np.cos(theta)

        xprev = loc_array[old_index,3]
        yprev = loc_array[old_index,4]
        zprev = loc_array[old_index,5]
        
        restriction = True
        while restriction:
            theta = 2 * np.pi * np.random.random_sample()
            phi = np.pi * np.random.random_sample()

            dx = Lbond * np.sin(phi) * np.cos(theta)
            dy = Lbond * np.sin(phi) * np.sin(theta)
            dz = Lbond * np.cos(phi)

            xx = xprev + dx
            yy = yprev + dy
            zz = zprev + dz

            new_loc = np.array([xx, yy, zz])

            Restriction = False

            # if (self.np_list.size > 0 ):
            #     if (rad is None or rad < 0):
            #         rad = 0
            #     checking_distances = np.linalg.norm(self.np_list[:,0:3] - new_loc, axis = 1) - (np_list[:,3] + rad )

            #     if checking_distances.min() < 0:
            #         Restriction = True
            #     else:  
            #         if (rad > 0):
            #             np.append(temp_locations, np.array([np.append(new_loc, rad)]))
            #         Restriction = False

            if (Restriction is False):
                Restriction = True
                if np.abs(zz) < self.L[2]/2. :
                    if monomer_increment == 1:
                        restriction = False
                    else:
                        xpp = loc_array[index-2,3]
                        ypp = loc_array[index-2,4]
                        zpp = loc_array[index-2,5]

                        dxp = xx - xpp
                        dyp = yy - ypp
                        dzp = zz - zpp

                        rpsq = dxp*dxp+dyp*dyp+dzp*dzp
                        rp = np.sqrt(rpsq)
                        if rp > rmin:
                            restriction = False
                    
                        if self.periodic == True:
                            if xx > self.xhi:
                                xx -= self.L[0]
                            if yy > self.yhi:
                                yy -= self.L[1]
                            if xx < self.xlo:
                                xx += self.L[0]
                            if yy < self.ylo:
                                yy += self.L[1]

        loc_array[index,3] = xx
        loc_array[index,4] = yy
        loc_array[index,5] = zz
        



    def add_diblock_rho0(self, part1, part2, frac, chl, rho0, Lbond, bond_type, rmin = 0.0):
        num_chns = int(self.L[0] * self.L[1] * self.L[2] * rho0/chl)
        self.add_diblock(part1, part2, frac, chl, num_chns, Lbond, bond_type, rmin)

    def add_diblock(self, part1, part2, frac, chl, num_chns, Lbond,bond_type, rmin = 0.0, rad = None):
        self.__add_particle_type(part1)
        self.__add_particle_type(part2)
        if (chl > 1): 
            self.__add_bond_type(bond_type)

        # resid = self.natoms + 1
        ns_loc = chl * num_chns
        xloc =  np.zeros((ns_loc, 6), 'd')

        nbonds_loc = num_chns * (chl - 1)
        bond_loc = np.empty((nbonds_loc,4), int)

        atom_id = self.natoms
        molecule_len = chl 


        self.__update_particle_count(molecule_len*num_chns)

        chn_id = self.num_chns
        self.num_chns += num_chns
        bond_count = 0

        for i_ch in range(num_chns):
            for i_monomer in range(chl):
                atom_id += 1

                tmp_index = i_ch * molecule_len + i_monomer
                if float(i_monomer)/float(chl) < frac:
                    xloc[tmp_index,2] = part1
                else:
                    xloc[tmp_index,2] = part2

                xloc[tmp_index, 0] = atom_id
                xloc[tmp_index, 1] = chn_id + i_ch
                if i_monomer == 0:
                    xloc[tmp_index, 3] = self.xlo + np.random.random_sample() * self.L[0]
                    xloc[tmp_index, 4] = self.ylo + np.random.random_sample() * self.L[1]
                    xloc[tmp_index, 5] = self.zlo + np.random.random_sample() * self.L[2]
                else:
                    bond_loc[bond_count, 0] = self.nbonds
                    bond_loc[bond_count, 1] = bond_type
                    bond_loc[bond_count, 2] = atom_id
                    bond_loc[bond_count, 3] = atom_id - 1 
                    bond_count += 1
                    self.nbonds += 1

                    self.__add_bond_check_bond_overlap(xloc, tmp_index, i_monomer, Lbond, rmin)


        self.x = np.concatenate([self.x, xloc])
        self.bonds = np.vstack([self.bonds, bond_loc])


    def add_homopolymer(self, part, chl, num_chns, Lbond, bond_type):
        self.add_diblock(part, part, 1.0, chl, num_chns, Lbond, bond_type)

    def add_np(self, part, num_part, radius):
        self.add_diblock(part, part, 1.0, chl, num_chns, Lbond=0, bond_type=None, rad = radius)

    def add_homopolymer_rho0(self, part, chl, rho0, Lbond, bond_type):
        num_chns = int(self.L[0] * self.L[1] * self.L[2] * rho0/chl)
        self.add_diblock(part, part, 1.0, chl, num_chns, Lbond, bond_type)
        

    def add_comb_rho0(self, bb_part1, Nb,Ns, rho0, ss_pt1, back_bond, bb_part2=None, frac_bb=1, ss_pt2=None,
            frac_side=1.0, Lbond=1.0, freq=1,
            back_side_bond=None, side_bond=None, rmin = 0.0):

        num_chns = int(self.L[0] * self.L[1] * self.L[2] * rho0 / (Nb + math.ceil(float(Nb)/freq ) * Ns))
        self.add_comb(bb_part1, Nb, Ns, num_chns, ss_pt1, back_bond, bb_part2=bb_part2, frac_bb=frac_bb, ss_pt2=ss_pt2,
                frac_side = frac_side, Lbond = Lbond, freq = freq, 
                back_side_bond = back_side_bond, side_bond = side_bond, rmin = rmin)

    def add_comb(self, bb_part1, Nb,Ns, num_chns, ss_pt1, back_bond, bb_part2=None, frac_bb=1, ss_pt2=None,
            frac_side=1.0, Lbond=1.0, freq=1,
            back_side_bond=None, side_bond=None, rmin = 0.0):
        self.__add_particle_type(bb_part1)
        self.__add_particle_type(bb_part2)
        self.__add_particle_type(ss_pt1)
        self.__add_particle_type(ss_pt2)
        
        self.__add_bond_type(back_bond)
        self.__add_bond_type(back_side_bond)
        self.__add_bond_type(side_bond)

        if side_bond == None:
            side_bond = back_bond
        if back_side_bond == None:
            back_side_bond = back_bond


        # resid = self.natoms + 1
        ns_loc = (Nb + Ns * Nb//freq) * num_chns
        xloc =  np.zeros((ns_loc, 6), 'd')

        old_natoms = self.natoms

        nbonds_loc = num_chns * ( (Nb - 1) + Nb//freq * (Ns) )
        bond_loc = np.empty((nbonds_loc,4), int)

        atom_id = self.natoms

        molecule_len =  Nb + Ns * Nb//freq

        self.__update_particle_count( molecule_len * num_chns)

        chn_id = self.num_chns
        self.num_chns += num_chns
        bond_count = 0

        for i_ch in range(num_chns):
            for i_monomer in range(Nb):
                atom_id += 1

                tmp_index = i_ch * molecule_len + i_monomer
                if float(i_monomer)/float(Nb) < frac_bb:
                    xloc[i_ch*molecule_len+i_monomer,2] = bb_part1
                else:
                    xloc[i_ch*molecule_len+i_monomer,2] = bb_part2

                xloc[tmp_index,0] = atom_id
                xloc[tmp_index,1] = chn_id + i_ch # molecule id 
                if i_monomer == 0:
                    xloc[tmp_index,3] = self.xlo + np.random.random_sample() * self.L[0]
                    xloc[tmp_index,4] = self.ylo + np.random.random_sample() * self.L[1]
                    xloc[tmp_index,5] = self.zlo + np.random.random_sample() * self.L[2]
                else:
                    bond_loc[bond_count, 0] = self.nbonds
                    bond_loc[bond_count, 1] = back_bond 
                    bond_loc[bond_count, 2] = atom_id - 1
                    bond_loc[bond_count, 3] = atom_id 

                    bond_count += 1
                    self.nbonds += 1

                    self.__add_bond_check_bond_overlap(xloc, tmp_index, i_monomer, Lbond, rmin)

            for i_monomer in range(Nb):
                for i_side in range(Ns): 
                    atom_id += 1

                    tmp_index = i_ch * molecule_len + Nb + i_monomer // freq * Ns + i_side
                    indbb = i_ch * molecule_len + i_monomer + 1
                    xloc[tmp_index,0] = atom_id 
                    xloc[tmp_index,1] = chn_id + i_ch # molecule id 
                    if float(i_side)/float(Ns) < frac_side:
                        xloc[tmp_index,2] = ss_pt1
                    else:
                        xloc[tmp_index,2] = ss_pt2

                    if i_side == 0:
                        bond_loc[bond_count, 0] = self.nbonds
                        bond_loc[bond_count, 1] = back_side_bond 
                        bond_loc[bond_count, 2] = indbb + old_natoms
                        # bond_loc[bond_count, 3] = tmp_index 
                        bond_loc[bond_count, 3] = atom_id
                        bond_count += 1
                        self.nbonds += 1
                        self.__add_bond_check_bond_overlap(xloc, tmp_index, i_side+1, Lbond, rmin, old_index = indbb-1 )

                    else:
                        bond_loc[bond_count, 0] = self.nbonds
                        bond_loc[bond_count, 1] = side_bond 
                        bond_loc[bond_count, 2] = atom_id- 1
                        bond_loc[bond_count, 3] = atom_id
                        bond_count += 1
                        self.nbonds += 1
                        self.__add_bond_check_bond_overlap(xloc, tmp_index, i_side+1, Lbond, rmin)

        self.x = np.concatenate([self.x, xloc])
        self.bonds = np.vstack([self.bonds, bond_loc])

    def add_simple_ABA_rho0(self,part1, part2, fracA, chl, rho0, Lbond=1.0, bond_type=1, rmin = 0.0):
        self.add_triblock(part1, part2, part1, fracA, 1-2*fracA, chl, int(self.L[0] * self.L[1] * self.L[2] * rho0/chl), Lbond = Lbond, 
                bond_type12 = bond_type, bond_type23= bond_type,rmin=rmin)

    def add_simple_ABA(self,part1, part2, fracA, chl, num_chns, Lbond=1.0, bond_type=1, rmin = 0.0):
        self.add_triblock(part1, part2, part1, fracA, 1-2*fracA, chl, num_chns, Lbond = Lbond, 
                bond_type12 = bond_type, bond_type23= bond_type,rmin=rmin)

    def add_triblock_rho0(self,part1, part2, part3, frac1, frac2, chl, rho0, Lbond=1.0, bond_type12=1, bond_type23=1, rmin = 0.0):
        self.add_triblock(part1, part2, part3, frac1, frac2, chl, int(self.L[0] * self.L[1] * self.L[2] * rho0/chl), Lbond = Lbond, 
                bond_type12 = bond_type12, bond_type23= bond_type23, rmin=rmin)

    def add_triblock(self, part1, part2, part3, frac1, frac2, chl, num_chns, Lbond=1.0,bond_type12=1, bond_type23=1, rmin = 0.0):
        self.__add_particle_type(part1)
        self.__add_particle_type(part2)
        self.__add_particle_type(part3)

        self.__add_bond_type(bond_type12)
        self.__add_bond_type(bond_type23)

        ns_loc = chl * num_chns
        xloc =  np.zeros((ns_loc, 6), 'd')

        nbonds_loc = num_chns * (chl - 1)
        bond_loc = np.empty((nbonds_loc,4), int)
        
        molecule_len = chl

        atom_id = self.natoms

        self.natoms += chl * num_chns
        chn_id = self.num_chns
        self.num_chns += chl
        bond_count = 0

        for i_ch in range(num_chns):
            for i_monomer in range(chl):
                atom_id += 1

                tmp_index = i_ch * molecule_len + i_monomer

                f_along = float(i_monomer)/float(chl) 
                if f_along < frac1:
                    xloc[tmp_index,2] = part1
                elif f_along < frac1+frac2:
                    xloc[tmp_index,2] = part2
                else:
                    xloc[tmp_index,2] = part3

                xloc[tmp_index,0] = atom_id 
                xloc[tmp_index,1] = chn_id + i_ch
                if i_monomer == 0:
                    xloc[tmp_index,3] = self.xlo + np.random.random_sample() * self.L[0]
                    xloc[tmp_index,4] = self.ylo + np.random.random_sample() * self.L[1]
                    xloc[tmp_index,5] = self.zlo + np.random.random_sample() * self.L[2]

                else:
                    if f_along >= frac1 + frac2:
                        bndtyp = bond_type23
                    else: 
                        bndtyp = bond_type12

                    bond_loc[bond_count, 0] = self.nbonds
                    bond_loc[bond_count, 1] = bndtyp
                    bond_loc[bond_count, 2] = atom_id
                    bond_loc[bond_count, 3] = atom_id - 1 

                    bond_count += 1
                    self.nbonds += 1

                    self.__add_bond_check_bond_overlap(xloc, tmp_index, i_monomer, Lbond, rmin)
        self.x = np.concatenate([self.x, xloc])
        self.bonds = np.vstack([self.bonds, bond_loc])



    def write(self, output):

        path = (os.path.dirname(os.path.abspath(output)))        
        os.makedirs(path, exist_ok=True)

        otp = open(output, 'w')
        otp.write("Generated by Chris' code\n\n")
        
        line = "%d atoms\n" % (self.natoms  )
        otp.write(line)
        line = "%d bonds\n" % len(self.bonds)
        otp.write(line)
        line = "%d angles\n" % (self.nangles)
        otp.write(line)
        # line = "%d dihedrals\n" % (self.ndihedrals)
        # otp.write(line)
        # line = "%d impropers\n" % (self.ndihedrals)
        # otp.write(line)
        line = "\n" 
        otp.write(line)

        line = "%d atom types\n" % len(self.masses)
        otp.write(line)
        line = "%d bond types\n" % len(self.bond_types)
        otp.write(line)
        line = "%d angle types\n" % self.nang_types
        otp.write(line)
        # line = "%d dihedral types\n" % self.ndihedrals
        # otp.write(line)
        # line = "%d improper types\n" % self.nimpropers
        # otp.write(line)
        line = "\n" 
        otp.write(line)

        line = '%f  %f xlo xhi\n' % (self.lo[0], self.hi[0])
        otp.write(line)
        line = '%f  %f ylo yhi\n' % (self.lo[1], self.hi[1])
        otp.write(line)
        line = '%f  %f zlo zhi\n\n' % (self.lo[2], self.hi[2])
        otp.write(line)
        
        if len(self.masses) > 0 :
            otp.write("Masses \n\n")
        for i, val in enumerate(self.masses):
                    line = "%d 1.000\n" % (val)
                    otp.write(line)

        otp.write("\nAtoms \n\n")

        for i, val in enumerate(self.x):
                        line = "{:d} {:d} {:d} {:f} {:f} {:f}\n" 
                        idx,m,t,x,y,z = val
                        otp.write(line.format(int(idx),int(m),int(t),x,y,z))

        if len(self.bonds) > 0 :
            otp.write("\nBonds \n\n")
        for i, val in enumerate(self.bonds):
            line = ' '.join(map(str, val))
            otp.write(line + "\n")

    def add_diblock_angle(self, part1, part2, frac, chl, num_chns, Lbond,bond_type,angle_type = None, rmin = 0.0):
        self.__add_particle_type(part1)
        self.__add_particle_type(part2)
        self.__add_bond_type(bond_type)

        resid = self.natoms + 1
        ns_loc = chl * num_chns
        xloc =  np.zeros((ns_loc, 6), 'd')
        # bond_loc = np.zeros((0, 4), 'd')
        # bond_loc = np.([], dtype=np.float).reshape(0,4)
        nbonds_loc = num_chns * (chl - 1)
        bond_loc = np.empty((nbonds_loc,4), int)
        nangles_loc = num_chns * (chl -2 )
        bond_loc = np.empty((nangles_loc,4), int)
        # self.nbonds 
        natoms = self.natoms
        self.natoms += chl * num_chns
        chn_id = self.num_chns
        self.num_chns += chl
        bond_count = 0
        if not angle_type == None:
            if ( not angle_type in self.ang_types):
                self.ang_types.append(part2)

        for i_ch in range(num_chns):
            for i_monomer in range(chl):
                natoms += 1
                if float(i_monomer)/float(chl) < frac:
                    xloc[i_ch*chl+i_monomer,2] = part1
                else:
                    xloc[i_ch*chl+i_monomer,2] = part2

                xloc[i_ch*chl+i_monomer,0] = natoms
                xloc[i_ch*chl+i_monomer,1] = chn_id + i_ch
                if i_monomer == 0:
                    xloc[i_ch*chl,3] = self.xlo + np.random.random_sample() * self.L[0]
                    xloc[i_ch*chl,4] = self.ylo + np.random.random_sample() * self.L[1]
                    xloc[i_ch*chl,5] = self.zlo + np.random.random_sample() * self.L[2]
                else:
                    bond_loc[bond_count, 0] = self.nbonds
                    bond_loc[bond_count, 1] = bond_type
                    bond_loc[bond_count, 2] = natoms
                    bond_loc[bond_count, 3] = natoms - 1 
                    bond_count += 1
                    self.nbonds += 1
                    theta = 2 * np.pi * np.random.random_sample()
                    phi = np.pi * np.random.random_sample()

                    dx = Lbond * np.sin(phi) * np.cos(theta)
                    dy = Lbond * np.sin(phi) * np.sin(theta)
                    dz = Lbond * np.cos(theta)

                    xprev = xloc[i_ch*chl+i_monomer-1,3]
                    yprev = xloc[i_ch*chl+i_monomer-1,4]
                    zprev = xloc[i_ch*chl+i_monomer-1,5]
                    

                    restriction = True
                    while restriction:
                        theta = 2 * np.pi * np.random.random_sample()
                        phi = np.pi * np.random.random_sample()

                        dx = Lbond * np.sin(phi) * np.cos(theta)
                        dy = Lbond * np.sin(phi) * np.sin(theta)
                        dz = Lbond * np.cos(phi)

                        xx = xprev + dx
                        yy = yprev + dy
                        zz = zprev + dz

                        if np.abs(zz) < self.L[2]/2. :
                            if i_monomer == 1:
                                restriction = False
                            else:
                                xpp = xloc[i_ch*chl+i_monomer-2,3]
                                ypp = xloc[i_ch*chl+i_monomer-2,4]
                                zpp = xloc[i_ch*chl+i_monomer-2,5]

                                dxp = xx - xpp
                                dyp = yy - ypp
                                dzp = zz - zpp

                                rpsq = dxp*dxp+dyp*dyp+dzp*dzp
                                rp = np.sqrt(rpsq)
                                if rp > rmin:
                                    restriction = False
                            
                                if self.periodic == True:
                                    if xx > self.xhi:
                                        xx -= self.L[0]
                                    if yy > self.yhi:
                                        yy -= self.L[1]
                                    if xx < self.xlo:
                                        xx += self.L[0]
                                    if yy < self.ylo:
                                        yy += self.L[1]

                    xloc[i_ch*chl+i_monomer,3] = xx
                    xloc[i_ch*chl+i_monomer,4] = yy
                    xloc[i_ch*chl+i_monomer,5] = zz
        self.x = np.concatenate([self.x, xloc])
        self.bonds = np.vstack([self.bonds, bond_loc])

