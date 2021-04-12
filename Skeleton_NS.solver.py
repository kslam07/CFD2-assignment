# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:25:58 2020

@author: mgerritsma

The numbering of the mesh is done as follows:
primal grid
- counting starts at the left bottom corner
- cells are first read from left to right, starting from the bottom to the top
- boundary cells are counted at the end, i.e. start at second left bottom cell
- fluxes / edges are counted starting from the most left bottom edge / flux
- flux at boundaries are known of course
- fluxes are counted in similar fashion: from bottom, left to right, to top
- After counting all the horizontal fluxes / vertical edges, the counting
  continues at the bottom left horizontal edge / vertical flux
- Points counting is in a similar fashion; there are no points in the boundary
  cells

dual grid
- Numbering of cells corresponds to number of points in primal grid
- Numbering of edges corresponds to numbering of edges in primal grid
- Numbering of points corresponds to numbering of cells in primal grid
"""

from scipy.sparse import diags
from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

#  00D#MMXX#

tol = float(1e-6)
one = int(1)
mone = int(-1)

L = float(1.0)
Re = float(1000)  # Reynolds number
N = int(16)  # mesh cells in x- and y-direction

u = np.zeros([2 * N * (N + 1), 1], dtype=np.float)  # circulation
p = np.zeros([N * N + 4 * N, 1], dtype=np.float)  # total pressure
tx = np.zeros([N + 1, 1], dtype=np.float)  # grid points on primal grid edges
x = np.zeros([N + 2, 1], dtype=np.float)  # grid points on dual grid sink/src
th = np.zeros([N], dtype=np.float)  # mesh width primal grid
h = np.zeros([N + 1], dtype=np.float)  # mesh width dual grid

# Generation of a non-uniform grid
x[0] = 0
x[N + 1] = 1
for i in range(N + 1):
    xi = i * L / N
    tx[i] = 0.5 * (1. - np.cos(np.pi * xi))  # tx mesh point for primal mesh
    if i > 0:
        th[i - 1] = tx[i] - tx[i - 1]  # th mesh width on primal mesh
        x[i] = 0.5 * (tx[i - 1] + tx[i])  # x mesh points for dual mesh

for i in range(N + 1):
    h[i] = x[i + 1] - x[i]  # h mesh width on dual mesh

th_min = min(th)
h_min = min(h)
h_min = min(h_min, th_min)  # determination smallest mesh size

dt = min(h_min, 0.5 * Re * h_min ** 2)  # dt for stable integration

dt = 5 * dt

run = True
#
#  Note that the time step is a bit conservative so it may be useful to see
#  if the time step can be slightly increased. This will speed up the
#  calculation.
#

#  Boundary conditions for the lid driven cavity test case
U_wall_top = -1
U_wall_bot = 0
U_wall_left = 0
U_wall_right = 0
V_wall_top = 0
V_wall_bot = 0
V_wall_left = 0
V_wall_right = 0


def get_idx_source(N):
    return N ** 2 + 4 * N


def get_idx_edge(N):
    return 2 * N * (N + 1)


def get_vertex_edge(N):
    return (N + 1) ** 2 + 4 * (N + 1)


def get_idx_circulation(N):
    return (N + 1) ** 2


def get_idx_edges_boundary(N):
    return 2 * N * (N + 3)
# in the assignment.
# Make sure to use sparse matrices to avoid memory problems


# RHS vector
def setup_tE10(N):
    N_edges = 2 * (N + 1) * N
    N_circ = (N + 1) ** 2
    rows = []
    cols = []
    data = []
    j1 = 0
    j2 = N + 1
    p = -1
    for i in range(2 * (N + 1) * N):
        if i < N * (N + 1):
            cols.extend([j1 + i, j2 + i])
            rows.extend([i, i])
            data.extend([-1, 1])
        else:
            if (i - N * (N + 1) + 1) % (N) == 1:
                p += 1
            cols.extend([i - N * (N + 1) + p, i - N * (N + 1) + 1 + p])
            rows.extend([i, i])
            data.extend([1, -1])
    # x=sparse.coo_matrix((data,(rows,cols)),shape=(N_edges, N_circ)).toarray()
    return (sparse.coo_matrix((data, (rows, cols)), shape=(N_edges, N_circ)))


def setup_E10(N):
    N_edges = get_idx_edge(N)
    N_vertices = get_idx_source(N)

    rows = []
    cols = []

    jb1 = N ** 2  # starting index for boundary pts starting from 0
    j = 0
    for i in range(int(N_edges / 2), N_edges):  # vertical edges
        if i < (int(N_edges / 2) + N):  # if line neighbours boundary pt
            cols.extend([j, jb1])
            jb1 += 1
        elif i >= (N_edges - N):  # if line neighbours boundary pt
            cols.extend([jb1, j - N])
            jb1 += 1
        else:  # looping through internal edges
            # # - N since internal points start after N iterations
            cols.extend([j, j - N])
        rows.extend([i, i])
        j += 1

    i = 1
    j = 1  # internal point starts after first boundary pt
    for __ in range(N * (N - 1)):  # horizontal internal edges
        cols.extend([j, j - 1])
        rows.extend([i, i])
        if (i + 2) % (N + 1) == 0:
            i += 3
            j += 2
        else:
            i += 1
            j += 1

    # add first (lower left) side edge
    cols.extend([0, jb1])
    rows.extend([0, 0])

    i = N
    jb1 += N
    for j in range(N - 1, N ** 2 - 1, N):  # side edges
        cols.extend([jb1, j, j + 1, jb1 - (N - 1)])
        rows.extend([i, i, i + 1, i + 1])
        i += N + 1
        jb1 += 1

    # add last (upper right) side edge
    cols.extend([jb1, N ** 2 - 1])
    rows.extend([i, i])

    return sparse.coo_matrix(([1, -1] * N_edges, (rows, cols)),
                             shape=(N_edges, N_vertices), dtype=np.int64)


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


def setup_tE21(N):
    cols = []
    rows = []
    data = []
    # inner cells
    for i in range(N ** 2):
        if i == 0:
            j1, j2, j3, j4 = 0, 1, (N + 1) * N, (N + 1) * N + N
        if i % N == 0 and i != 0:
            j1, j2, j3, j4 = j1 + 2, j2 + 2, j3 + 1, j4 + 1
        elif i != 0:
            j1, j2, j3, j4 = j1 + 1, j2 + 1, j3 + 1, j4 + 1
        # print([j1,j2,j3,j4])
        cols.extend([j1, j2, j3, j4])
        rows.extend([i, i, i, i])
        data.extend([-1, 1, -1, 1])
    # boundary points

    # bottom boundary points
    for j in range(N):
        i += 1
        if j == 0:
            j1 = (N + 1) * N
            cols.extend([j1])
            rows.extend([i])
            data.extend([1])
        else:
            cols.extend([j1 + j])
            rows.extend([i])
            data.extend([1])

    # top row
    for j in range(N):
        i += 1
        if j == 0:
            j1 = (N + 1) * N * 2 - (N)
            cols.extend([j1])
            rows.extend([i])
            data.extend([-1])
        else:
            cols.extend([j1 + j])
            rows.extend([i])
            data.extend([-1])

    # left
    for j in range(N):
        i += 1
        if j == 0:
            j1 = 0
            cols.extend([j1])
            rows.extend([i])
            data.extend([1])
        else:
            cols.extend([j1 + j * (N + 1)])
            rows.extend([i])
            data.extend([1])

    # left
    for j in range(N):
        i += 1
        if j == 0:
            j1 = N
            cols.extend([j1])
            rows.extend([i])
            data.extend([-1])
        else:
            cols.extend([j1 + j * (N + 1)])
            rows.extend([i])
            data.extend([-1])

    # x=sparse.coo_matrix((data,(rows,cols)),shape=(21,36)).toarray()
    return sparse.coo_matrix((data, (rows, cols)),
                             shape=(N ** 2 + 4 * N, 2 * (N + 1) * N),
                             dtype=np.int64)


def setup_Ht11(N, th, h):
    # initialize lists
    rows = []
    data = []

    # compute number of edges
    idx_max_edge = get_idx_edge(N)
    idx_max_edgeh = int(idx_max_edge / 2) - 1

    # list of th repeats
    th_lst_h = np.repeat(th, N + 1)
    h_lst_h = h.tolist() * N  # [a, b, c, a, b, c, ...]
    h_lst_v = np.repeat(h, N)  # [a, a, ..., b, b, ..., c, c, ...]
    th_lst_v = th.tolist() * (N + 1)
     # list of indices
    idx_h_lst = np.arange(0, idx_max_edgeh + 1)
    idx_v_lst = np.arange(idx_max_edgeh + 1, idx_max_edge)

    for ht_i, h_i, idx_h in zip(th_lst_h, h_lst_h, idx_h_lst):
        hrat = ht_i / h_i
        data.append(hrat)
        rows.append(idx_h)
    for ht_i, h_i, idx_v in zip(th_lst_v, h_lst_v, idx_v_lst):
        hrat = ht_i / h_i
        data.append(hrat)
        rows.append(idx_v)

    return sparse.coo_matrix((data, [rows, rows]),
                             dtype=np.float64,
                             shape=(idx_max_edge, idx_max_edge))


def setup_H1t1(Ht11):
    return sparse.linalg.inv(Ht11.tocsc()).tocoo()


def setup_Ht02(N, h):
    data = []
    cols = []
    for i, (hv, hh) in enumerate(product(h, h)):
        data.append(1 / (hv * hh))
        cols.append(i)

    return sparse.coo_matrix((data, [cols, cols]),
                             dtype=np.float64,
                             shape=((N + 1) ** 2, (N + 1) ** 2))


def Et21_rhs_mat(N):
    length = get_idx_source(N)
    rhs = np.zeros((length, 2 * N * (N + 3)))
    sign = -1
    idx = N ** 2 - 1
    idx_edge = 2 * N * (N + 1) - 1
    for i in range(4):
        sign *= -1
        for j in range(1, N + 1):
            idx += 1
            idx_edge += 1
            rhs[idx][idx_edge] = sign
    return rhs

tE21 = setup_tE21(N)
E21, u_pres = setup_E21(N, U_wall_top, U_wall_bot, V_wall_left, V_wall_right, h)
E10 = -tE21.T
tE10 = E21.T
u_norm = Et21_rhs_mat(N) @ np.zeros((get_idx_edges_boundary(N), 1))
Ht11 = setup_Ht11(N, th, h)
H1t1 = setup_H1t1(Ht11)
Ht02 = setup_Ht02(N, h)

# Au = RHS

if run:
    A = tE21 @ Ht11 @ E10

    n = A.shape[0]
    LU = splinalg.splu(A.tocsc(), diag_pivot_thresh=0)  # sparse LU decomposition

    u_pres_vort = Ht02 @ u_pres
    temp = H1t1 @ tE10 @ Ht02 @ u_pres

    u_pres = temp

    VLaplace = H1t1 @ tE10 @ Ht02 @ E21
    DIV = tE21 @ Ht11

    ux_xi = np.zeros([(N + 1) * (N + 1), 1], dtype=float)
    uy_xi = np.zeros([(N + 1) * (N + 1), 1], dtype=float)
    convective = np.zeros([2 * N * (N + 1), 1], dtype=float)

    diff = 1
    iter = 1

    while (diff > tol):

        xi = Ht02 @ E21 @ u + u_pres_vort

        for i in range(N + 1):
            for j in range(N + 1):
                k = j + i * (N + 1)
                if j == 0:
                    ux_xi[k] = U_wall_bot * xi[i + j * (N + 1)]
                    uy_xi[k] = V_wall_left * xi[j + i * (N + 1)]
                elif j == N:
                    ux_xi[k] = U_wall_top * xi[i + j * (N + 1)]
                    uy_xi[k] = V_wall_right * xi[j + i * (N + 1)]
                else:
                    ux_xi[k] = (u[i + j * (N + 1)] + u[i + (j - 1) * (N + 1)]) * \
                               xi[i + j * (N + 1)] / (2. * h[i])  # Klopt
                    uy_xi[k] = (u[N * (N + 1) + j + i * N] + u[
                        N * (N + 1) + j - 1 + i * N]) * xi[j + i * (N + 1)] / (
                                           2. * h[i])

        for i in range(N):
            for j in range(N + 1):
                convective[j + i * (N + 1)] = -(uy_xi[j + i * (N + 1)] + uy_xi[
                    j + (i + 1) * (N + 1)]) * h[j] / 2.
                convective[N * (N + 1) + i + j * N] = (ux_xi[j + i * (N + 1)] +
                                                       ux_xi[j + (i + 1) * (
                                                                   N + 1)]) * h[
                                                          j] / 2.

        # Set up the right hand side for the equation for the pressure

        rhs_Pois = DIV @ (
                    u / dt - convective - VLaplace @ u / Re - u_pres / Re) +\
                   u_norm / dt

        # Solve for the pressure

        p = LU.solve(rhs_Pois)

        # Store the velocity from the previous time level in the vector uold

        uold = u

        # Update the velocity field

        u = u - dt * (convective + E10 @ p + (VLaplace @ u) / Re + u_pres / Re)

        # Every other 1000 iterations check whether you approach steady state and
        # check whether you satsify conservation of mass. The largest rate at whci
        # mass is created ot destroyed is denoted my 'maxdiv'. This number should
        # be close to machine precision.

        if ((iter % 1000) == 0):
            maxdiv = max(abs(DIV @ u + u_norm))
            diff = max(abs(u - uold)) / dt

            print("maxdiv : ", maxdiv)
            print("diff   : ", diff)

        iter += 1


def test_matrices(E10, E21, tE10, tE21, H1t1, Ht02, Ht11):

    E10 = E10.toarray()
    E21 = E21.toarray()
    tE10 = tE10.toarray()
    tE21 = tE21.toarray()
    H1t1 = H1t1.toarray()
    Ht02 = Ht02.toarray()
    Ht11 = Ht11.toarray()

    return E10, E21, tE10, tE21, H1t1, Ht02, Ht11


def test_E10(E10, tE21):
    return np.allclose(-E10.T, tE21)


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


def plot_contour(N, x, u_org, p_org, h):
    p = compute_pstat(N, p_org, u_org, h)
    # internal grid
    X, Y = np.meshgrid(x[1:-1], x[1:-1])
    prescont = [-0.002,0.0,0.02,0.05,0.07,0.09,0.11,0.12,0.17,0.3]
    fig, ax = plt.subplots(1, 1, dpi=150)
    cs = ax.contour(X, Y, p, levels=prescont)
    ax.clabel(cs, inline=1, fontsize=12)
    ax.set_xlabel("x [-]")
    ax.set_ylabel("y [-]")

    return fig, ax


def plot_streamfunction(N, u, x):
    X, Y = np.meshgrid(x, x)

    phi = sparse.linalg.lsqr(tE10, tu)[0].reshape(N + 1, N + 1)
    # -1.5e-3, -1e-3, -5e-4, -2.5e-4, -1e-4, -5e-5, -1e-5, -1e-6,
    lvls = [-1.5e-3, -1e-3, -5e-4, -2.5e-4, -1e-4, -5e-5, -1e-5, -1e-6, 0.0, 1e-10, 1e-5, 1e-4, 1e-2, 3e-2, 5e-2, 7e-2,
            9e-2, 0.1, 0.11, 0.115, 0.1175]
    # lvls = [0.115, 0.1175]
    fig, ax = plt.subplots(1, 1, dpi=150)
    cs = ax.contour(X, Y, phi[::-1], levels=lvls)
    ax.clabel(cs, inline=1, fontsize=12)
    ax.set_xlabel("x [-]")
    ax.set_ylabel("y [-]")


def plot_vorticity(N, Ht02, E21, u_org, u0):

    u = u_org.copy()
    # compute grid
    X, Y = np.meshgrid(tx, tx)
    # compute vorticity
    vort = Ht02 @ E21 @ u + u0

    # plot vorticity
    fig, ax = plt.subplots(1, 1, dpi=150)
    levels = [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0][::-1]
    cs = ax.contour(X, Y, vort.reshape(N + 1, N + 1), levels=levels)

plot_contour(N, x, u, p, h)
plot_vorticity(N, Ht02, E21, u, u_pres_vort)
plot_streamfunction(N, u, tx)
# TODO plot velocity at given x OR y
