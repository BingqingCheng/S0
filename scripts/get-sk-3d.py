#!/usr/bin/python

import numpy as np
import glob,sys
from math import pi
import time

def read_lammpstrj(filedesc):
    # three comment lines
    for i in range(3): comment = filedesc.readline()
    # number of atoms
    natoms = int(filedesc.readline())
    #print(natoms)
    # 1 comment line
    comment = filedesc.readline()
    # assume orthorombic cell
    cell = np.zeros(3,float)
    for i in range(3): 
        [cellmin, cellmax] = filedesc.readline().split()
        cell[i] = float(cellmax) - float(cellmin)
    # 1 comment line
    comment = filedesc.readline()
    names = np.zeros(natoms,'U2')
    q = np.zeros((natoms,3),float)
    sq = np.zeros((natoms,3),float)

    for i in range(natoms):
        line = filedesc.readline().split();
        names[i] = line[0] # atom type
        q[i] = line[1:4] # wrapped atomic coordinates
        sq[i,0] = float(q[i,0])/cell[0] # scaled atomic coordinates
        sq[i,1] = float(q[i,1])/cell[1] # scaled atomic coordinates
        sq[i,2] = float(q[i,2])/cell[2] # scaled atomic coordinates
    #print(names)
    return [cell, names, sq]

def Sk(names, q, kgrid, e_A, e_B):
    # This is the un-normalized FT for the density fluctuations
    q_A = np.asarray([ q_now for i,q_now in enumerate(q) if names[i] in e_A ])
    n_A = len(q_A)
    print("Number of element A: ", n_A)
    if n_A > 0:
        FTrho_A = FT_density(q_A, kgrid)
    else:
        FTrho_A = np.empty(len(kgrid))
        FTrho_A[:] = np.NaN
    if e_A != e_B:
        q_B = np.asarray([ q_now for i,q_now in enumerate(q) if names[i] in e_B ])
        n_B = len(q_B)
        print("Number of element B: ", n_B)
        if n_B > 0:
            FTrho_B = FT_density(q_B, kgrid)
        else:
            FTrho_B = np.empty(len(kgrid))
            FTrho_B[:] = np.NaN
    else:
        FTrho_B = FTrho_A

    return np.multiply(FTrho_A, np.conjugate(FTrho_A))/n_A, \
                   np.multiply(FTrho_A, np.conjugate(FTrho_B))/(n_A*n_B)**0.5, \
                   np.multiply(FTrho_B, np.conjugate(FTrho_B))/n_B

def FT_density(q, kgrid):
    # This is the un-normalized FT for density fluctuations
    ng = len(kgrid)
    ak = np.zeros(ng,dtype=complex)

    for n,k in enumerate(kgrid):
        ak[n] = np.sum(np.exp(-1j*(q[:,0]*k[0]+q[:,1]*k[1]+q[:,2]*k[2])))
    return ak

def main(sprefix="Sk", straj="out", sbins=8):

    # the input file
    print("Reading file:", straj,".lammpstrj")
    traj = open(straj+'.lammpstrj',"r")
    # number of k grids
    bins = int(sbins)
    # get total number of bins and initialize the grid
    print("Use number of bins:", bins)

    # Outputs
    ofile_AA = open(sprefix+'-II-real.dat',"ab")
    ofile_AB = open(sprefix+'-IW-real.dat',"ab")
    ofile_BB = open(sprefix+'-WW-real.dat',"ab")

    nframe = 0
    while True:
        start_time = time.time()
        # read frame
        try:
            [ cell, names, sq] = read_lammpstrj(traj)
        except:
            break
        nframe += 1
        print("Frame No:", nframe)

        if (nframe == 1):
            # normalization
            volume = np.prod(cell[:])

            kgrid = np.zeros((bins*bins*bins,3),float)
            kgridbase = np.zeros((bins*bins*bins,3),float)
            # initialize k grid
            [ dkx, dky, dkz ] = [ 1./cell[0], 1./cell[1], 1./cell[2] ]
            n=0
            for i in range(bins):
                for j in range(bins):
                    for k in range(bins):
                        if i+j+k == 0: pass
                        # initialize k grid
                        kgridbase[n,:] = (2.*pi)*np.array([i, j, k])
                        kgrid[n,:] = [ dkx*i, dky*j, dkz*k ]
                        n+=1
            np.savetxt(sprefix+'-kgrid.dat',kgrid)
        print("--- %s seconds after read frame ---" % (time.time() - start_time))
        # FT analysis of density fluctuations
        sk_AA, sk_AB, sk_BB = Sk(names, sq, kgridbase, ['Na','Cl'], ['O'])
        print("--- %s seconds after FFT density ---" % (time.time() - start_time))

        # Outputs
        np.savetxt(ofile_AA,sk_AA[None].real, fmt='%4.4e', delimiter=' ',header="Frame No: "+str(nframe))
        np.savetxt(ofile_AB,sk_AB[None].real, fmt='%4.4e', delimiter=' ',header="Frame No: "+str(nframe))
        np.savetxt(ofile_BB,sk_BB[None].real, fmt='%4.4e', delimiter=' ',header="Frame No: "+str(nframe))

    print("A total of data points ", nframe)

    sys.exit()


if __name__ == '__main__':
    main(*sys.argv[1:])

# to use: python ./get-sk-3d.py [inputfile] [outputfile] [nbin]
