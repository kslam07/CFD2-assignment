# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 20:18:55 2021

@author: Thomas Verduyn
"""

import pickle
import numpy as np
from scipy.sparse import diags
from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from itertools import product


results=pickle.load(open('resultsN31.pickle','rb'))

N=15
p=results[N]['p']
u=results[N]['u']
h=results[N]['h']
u_pres=results[N]['u_pres']

def get_idx_edge(N): # N is number of cells
    return 2 * N * (N + 1)

def get_idx_circulation(N):
    return (N + 1) ** 2

def setup_Ht02(N, h):
    data = []
    cols = []
    for i, (hv, hh) in enumerate(product(h, h)):
        data.append(1 / (hv * hh))
        cols.append(i)

    return sparse.coo_matrix((data, [cols, cols]),
                              dtype=np.float64,
                              shape=((N + 1) ** 2, (N + 1) ** 2))

def setup_E21(N, U_wall_top, U_wall_bot, V_wall_left, V_wall_right, h):
    idx_max_circ = get_idx_circulation(N)
    idx_max_edges = get_idx_edge(N)
    idx_start_vert_edge = int(idx_max_edges / 2)

    cols = []
    rows = []
    data = []
    rhs = np.zeros((idx_max_circ, 1))

    # edges
    # lower left, lower right, upper left, upper right
    cols.extend([idx_start_vert_edge, 0,
                  N, idx_start_vert_edge + N - 1,
                  idx_max_edges - N, idx_start_vert_edge - N - 1,
                  idx_max_edges - 1, idx_start_vert_edge - 1])
    rows.extend([0, 0, N, N, idx_max_circ - N - 1, idx_max_circ - N - 1,
                  idx_max_circ - 1, idx_max_circ - 1])
    data.extend([1, -1, -1, -1, 1, 1, -1, 1])

    # RHS edges
    rhs[0] = V_wall_left - U_wall_bot
    rhs[N] = -V_wall_right - U_wall_bot
    rhs[idx_max_circ - N - 1] = (U_wall_top + V_wall_left) * h[-1]
    rhs[idx_max_circ - 1] = (U_wall_top - V_wall_right) * h[-1]

    # bottom boundary cells
    jv = idx_start_vert_edge
    for i in range(1, N):
        cols.extend([jv + 1, i, jv])
        rows.extend([i] * 3)
        data.extend([1, -1, -1])
        rhs[rows[-1]] = -U_wall_bot
        jv += 1  # update iterands

    # top boundary cells
    jv = idx_max_edges - 1
    jh = idx_start_vert_edge - 2  # skipping upper right corner
    for i_rhs, i in enumerate(range(idx_max_circ - 1, idx_max_circ - N, -1)):
        cols.extend([jv, jv - 1, jh])
        rows.extend([i - 1] * 3)  # -1 due to Python
        data.extend([1, -1, 1])
        rhs[rows[-1]] = U_wall_top * h[i_rhs+1]
        jv -= 1
        jh -= 1

    # left and right boundary cells
    i = N + 1  # + 1 for Python
    jv = idx_start_vert_edge + N
    jh1 = 0
    jh2 = i
    for __ in range(1, N):
        cols.extend([jv, jh2, jh1,
                      jh2 + N, jv + N - 1, jh1 + N])
        rows.extend([i, i, i,
                      i + N, i + N, i + N])
        data.extend([1, -1, 1, -1, -1, 1])
        rhs[i] = V_wall_left
        rhs[i + N] = -V_wall_right

        i += N + 1
        jv += N
        jh1 += N + 1
        jh2 += N + 1

    # internal cells
    i = N + 2  # first internal point
    jv = idx_start_vert_edge + N
    jh = 1

    for idx in range(1, (N - 1) ** 2 + 1):
        cols.extend([jv + 1, jh + N + 1, jv, jh])
        rows.extend([i] * 4)
        data.extend([1, -1, -1, 1])

        if idx % (N - 1) == 0:  # go to next row (+2 due to side boundaries)
            jh += 2
            jv += 1
            i += 2

        # update iterands
        jh += 1
        jv += 1
        i += 1

    return sparse.coo_matrix((data, (rows, cols)),
                              shape=(idx_max_circ, idx_max_edges),
                              dtype=np.int64), -rhs
    
  
Ht02 = setup_Ht02(N, h)
U_wall_top = -1
U_wall_bot = 0
U_wall_left = 0
U_wall_right = 0
V_wall_top = 0
V_wall_bot = 0
V_wall_left = 0
V_wall_right = 0
E21, u_pres = setup_E21(N, U_wall_top, U_wall_bot, V_wall_left, V_wall_right, h)
tx = np.zeros([N + 1, 1], dtype=np.float)
x = np.zeros([N + 2, 1], dtype=np.float)
L=1.0
x[0] = 0
x[N + 1] = 1
for i in range(N + 1):
    xi = i * L / N
    tx[i] = 0.5 * (1. - np.cos(np.pi * xi))  # tx mesh point for primal mesh
    if i > 0:
        x[i] = 0.5 * (tx[i - 1] + tx[i])  # x mesh points for dual mesh
u_pres_vort = Ht02 @ u_pres

def compute_pstat(N, p_org, u_org, h_org):
    idx_edges = get_idx_edge(N)

    p_tot = p_org[:N**2].copy().reshape(N, N)
    vel = u_org.copy()
    h = h_org.reshape(-1, N+1)

    u = vel[:int(idx_edges/2)].reshape(N, N+1) / h  # reshape u to mesh shape
    v = vel[int(idx_edges/2):].reshape(N+1, N) / h.T

    # neighbouring rows neighbour a point for u
    u_magn = (u[:, :-1] + u[:, 1:]) / 2
    v_magn = (v[:-1, :] + v[1:, :]) / 2

    vel_magn = np.sqrt(u_magn**2 + v_magn**2)

    p = p_tot - 0.5 * vel_magn**2

    # take a reference pressure; taken in the centre of mesh
    if N % 2:  # then odd, there must be a centre point
        p_ref = p[int(N/2), int(N/2)]
    else:
        p_ref = np.mean(p[[int(N/2)-1, int(N/2)], :][:, [int(N/2)-1, int(N/2)]])

    p = p - p_ref

    return p


def plot_centervals(N,u_org,p_org,h,u0):
    p = compute_pstat(N, p_org, u_org, h)
    dataVert=np.loadtxt('vertical_data.txt',skiprows=1)
    dataHori=np.loadtxt('horizontal_data.txt',skiprows=1)
    pVert=np.array([])
    pHor=np.array([])
    if (len(p[0,:])%2)==0:
        for idx in range(len(p[0,:])):
            pVert=np.append(pVert,np.mean((p[idx,int(len(p[0,:])/2)-1],p[idx,int(len(p[0,:])/2)]),dtype=np.float64))    
            pHor=np.append(pHor,np.mean((p[int(len(p[:,0])/2)-1,idx],p[int(len(p[:,0])/2),idx]),dtype=np.float64))
    else:
        pVert=p[:,int(len(p)/2)]
        pHor=p[int(len(p)/2)]
    
    # Vortices
    vort = (Ht02 @ E21 @ u_org + u0).reshape(N+1,N+1)
    if (len(p[0,:])%2)==0:
        vortH=vort[int(len(vort)/2),:]
        vortV=vort[:,int(len(vort)/2)]
    else:
        vortH=np.array(np.mean(np.array([ vort[int(len(vort)/2)], vort[int(len(vort)/2)+1] ]), axis=0 ))
        vortV=np.array(np.mean(np.array([ vort[:,int(len(vort)/2)], vort[:,int(len(vort)/2)+1] ]), axis=0))
        
    # Velocity
    v=np.array([])
    u=np.array([])
    if (len(p[0,:])%2)==0: # odd numbers, can take average 
        v=u_org[np.arange(int(N/2)+1,int(get_idx_edge(N)/2),N+1)]
        u=u_org[np.arange(int(get_idx_edge(N)/2)+int(N/2)+1,get_idx_edge(N),N)]
    else: # even N, must average
        for idx in range(N):
            i=np.arange(int(N/2)+1,int(get_idx_edge(N)/2),N+1)
            v=np.mean((u_org[i],u_org[i-1]),axis=0)
            i=np.arange(int(get_idx_edge(N)/2)+int(N/2)+1,get_idx_edge(N),N)
            u=np.mean((u_org[i],u_org[i-1]),axis=0)
            
    idx_edges = get_idx_edge(N)
    hre = h.reshape(-1, N+1)
    u = u_org[:int(idx_edges/2)].reshape(N, N+1) / hre  # reshape u to mesh shape
    v = u_org[int(idx_edges/2):].reshape(N+1, N) / hre.T

    umid=np.array([])
    vmid=np.array([])
    if (len(u)%2)==0:
        umid=np.array(u[:,int(len(u[0])/2)])
        vmid=np.array(v[int(len(v[0])/2)])
    else:
        umid=np.array(np.mean(np.array([u[:,int(len(u[0])/2)],u[:,int(len(u[0])/2)+1]]),axis=0))
        vmid=np.array(np.mean(np.array([v[int(len(v[0])/2),:],v[int(len(v[0])/2)+1,:]]),axis=0))


    
    fig, ax = plt.subplots(3,2)
    ax[0,0].plot(dataVert[:,0],dataVert[:,3])
    ax[0,0].plot(((np.roll(tx,-1)+tx)/2)[:-1],pVert)
    ax[0,1].plot(dataHori[:,0],dataHori[:,3])
    ax[0,1].plot(((np.roll(tx,-1)+tx)/2)[:-1],pHor)
    
    ax[1,0].plot(dataVert[:,0],dataVert[:,-1])
    ax[1,0].plot(tx,vortV)
    ax[1,1].plot(dataHori[:,0],dataHori[:,-1])
    ax[1,1].plot(tx,vortH)
    
    # vel = u_org.copy()
    # h = h.reshape(-1, N+1)

    # u = vel[:int(idx_edges/2)].reshape(N, N+1) / h  # reshape u to mesh shape
    # v = vel[int(idx_edges/2):].reshape(N+1, N) / h.T

    ax[2,0].plot(dataVert[:,0],dataVert[:,2])
    ax[2,0].plot(((np.roll(tx,-1)+tx)/2)[:-1],umid)
    ax[2,1].plot(dataHori[:,0],dataHori[:,2])
    ax[2,1].plot(((np.roll(tx,-1)+tx)/2)[:-1],vmid)
    
    ax[0,0].grid()
    ax[0,1].grid()
    ax[1,0].grid()
    ax[1,1].grid()
    ax[2,0].grid()
    ax[2,1].grid()
    ax[0,0].set_xlabel('y')
    ax[0,1].set_xlabel('x')
    ax[1,0].set_xlabel('y')
    ax[1,1].set_xlabel('x')
    ax[2,0].set_xlabel('y')
    ax[2,1].set_xlabel('x')
    ax[0,0].set_ylabel('p')
    ax[0,1].set_ylabel('p')
    ax[1,0].set_ylabel(r'$\xi$')
    ax[1,1].set_ylabel(r'$\xi$')
    ax[2,0].set_ylabel('u')
    ax[2,1].set_ylabel('v')
    ax[0,0].set_title('Pressure Vertical')
    ax[0,1].set_title('Pressure Horizontal')
    ax[1,0].set_title('Vorticity Vertical')
    ax[1,1].set_title('Vorticity Horizontal')
    ax[2,0].set_title('Horizontal Velocity $u$ along $y$')
    ax[2,1].set_title('Vertical Velocity $y$ along $x$')
    ax[0,0].legend(['Botella','Solver'])
    plt.tight_layout()
    plt.show()
plot_centervals(N, u, p, h,u_pres_vort)