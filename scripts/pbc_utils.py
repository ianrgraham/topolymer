import numpy as nmp
import math as m


def pbc_vdr(x1, x2, box, boxh):
    dr = x1 - x2

    while (dr[0] >= boxh[0]):
        dr[0] -= box[0]
    while (dr[0] < -boxh[0]):
        dr[0] += box[0]

    while (dr[1] >= boxh[1]):
        dr[1] -= box[1]
    while (dr[1] < -boxh[1]):
        dr[1] += box[1]

    while (dr[2] >= boxh[2]):
        dr[2] -= box[2]
    while (dr[2] < -boxh[2]):
        dr[2] += box[2]

    return dr

def pbc_mdr(x1, x2, box, boxh):
    dr = pbc_vdr(x1,x2,box,boxh)

    mdr = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]

    return m.sqrt(mdr)

def pbc_pos_inbox(x1, box):
    if (x1[0] >= box[0]):
        x1[0] -= box[0]
    elif (x1[0] < 0.0):
        x1[0] += box[0]

    if (x1[1] >= box[1]):
        x1[1] -= box[1]
    elif (x1[1] < 0.0):
        x1[1] += box[1]

    if (x1[2] >= box[2]):
        x1[2] -= box[2]
    elif (x1[2] < 0.0):
        x1[2] += box[2]

    return x1

def pbc_inbox(x1, box, boxh):
    if (x1[0] >= boxh[0]):
        x1[0] -= box[0]
    elif (x1[0] < -boxh[0]):
        x1[0] += box[0]

    if (x1[1] >= boxh[1]):
        x1[1] -= box[1]
    elif (x1[1] < -boxh[1]):
        x1[1] += box[1]

    if (x1[2] >= boxh[2]):
        x1[2] -= box[2]
    elif (x1[2] < -boxh[2]):
        x1[2] += box[2]

    return x1


def pbc_mdr2(x1, x2, box, boxh):
    dr = pbc_vdr(x1,x2,box,boxh)

    mdr2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]

    return mdr2


def pbc_nojumps(x,bx):
    ns = nmp.shape(x)[1]
    fr = nmp.shape(x)[0]

    bxh = bx * 0.5

    for t in range(1,fr):
        for i in range(0,ns):
            dr = pbc_vdr(x[t-1,i], x[t,i], bx[t], bxh[t])  #x[t-1] - x[t]
            x[t] = x[t-1] - dr

    return x



def guess_box(x):

    # If "x" contains a trajectory
    if ( len(nmp.shape(x)) == 3):
        fr = nmp.shape(x)[0]
        bx = nmp.zeros((fr,3),'d')
        for i in range(0,fr):
            for j in range(0,3):
                bx[i,j] = nmp.max(x[i,:,j]) - nmp.min(x[i,:,j])

    # If "x" is a single frame
    else:
        bx = nmp.zeros(3,'d')
        for j in range(0,3):
            bx[j] = nmp.max(x[:,j]) - nmp.min(x[:,j])


    return bx
