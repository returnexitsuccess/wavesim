#!/usr/bin/env python3

# Code adapted from
# Finite difference methods for wave equations
# by Langtangen and Linge

import time
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Solve wave equation Dt Dt u = D (c(x,y)^2 D u) + f
# in 2D rectangular domain on mesh grid with initial conditions I, V
# and Dirichlet or Neumann boundary conditions
def solver(I, V, f, c, bds, Lx, Ly, T, dt, Nx, Ny, callback=None):
    Nt = int(round(T / dt))
    ts = np.linspace(0, Nt * dt, Nt + 1)

    xs = np.linspace(0, Lx, Nx + 1).reshape((Nx + 1, 1))
    ys = np.linspace(0, Ly, Ny + 1).reshape((1, Ny + 1))

    dt = ts[1] - ts[0]
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    if isinstance(c, (float, int)):
        q = np.full((Nx + 1, Ny + 1), c * c, dtype='float64')
    elif callable(c):
        cs = np.vectorize(c)(xs, ys).astype('float64')
        q = cs * cs
    dt2 = dt * dt
    Cx2 = dt2 / (dx * dx)
    Cy2 = dt2 / (dy * dy)

    bd_x0, bd_xL, bd_y0, bd_yL = bds

    # allow ease of setting initial conditions or forcing term
    if f is None or f == 0:
        f = lambda x, t: 0
    if V is None or V == 0:
        V = lambda x: 0
    if bd_x0 == 0:
        bd_x0 = lambda y, t: 0
    if bd_xL == 0:
        bd_xL = lambda y, t: 0
    if bd_y0 == 0:
        bd_y0 = lambda x, t: 0
    if bd_yL == 0:
        bd_yL = lambda x, t: 0

    f = np.vectorize(f)
    V = np.vectorize(V)
    if bd_x0 is not None:
        bd_x0 = np.vectorize(bd_x0)
    if bd_xL is not None:
        bd_xL = np.vectorize(bd_xL)
    if bd_y0 is not None:
        bd_y0 = np.vectorize(bd_y0)
    if bd_yL is not None:
        bd_yL = np.vectorize(bd_yL)

    # initialize u arrays with ghost cells
    u = np.zeros((Nx + 3, Ny + 3)) # u array at new timestep
    u_n = np.zeros((Nx + 3, Ny + 3)) # u array at current timestep
    u_nm1 = np.zeros((Nx + 3, Ny + 3)) # u array at previous timestep

    t0 = time.process_time()

    # initial condition
    u_n[1:-1,1:-1] = np.vectorize(I)(xs, ys).astype('float64')

    if callback is not None:
        callback(u_n[1:-1], xs, ys, ts, 0)

    # special first timestep formula
    u[1:-1,1:-1] = u_n[1:-1,1:-1] \
        + dt * V(xs[1:-1], ys[1:-1]) \
        + 0.25 * Cx2 * ((q[1:-1,1:-1] + q[2:,1:-1]) * (u_n[2:,1:-1] - u_n[1:-1,1:-1]) - (q[1:-1,1:-1] + q[:-2,1:-1]) * (u_n[1:-1,1:-1] - u_n[:-2,1:-1])) \
        + 0.25 * Cy2 * ((q[1:-1,1:-1] + q[1:-1,2:]) * (u_n[1:-1,2:] - u_n[1:-1,1:-1]) - (q[1:-1,1:-1] + q[1:-1,:-2]) * (u_n[1:-1,1:-1] - u_n[1:-1,:-2])) \
        + 0.5 * dt2 * f(xs[1:-1], ys[1:-1], 0)
    if bd_x0 is None:
        # Neumann
        u[0,:] = u[2,:]
    else:
        # Dirichlet
        u[1,:] = bd_x0(ys, ts[1])
    if bd_xL is None:
        # Neumann
        u[-1,:] = u[-3,:]
    else:
        # Dirichlet
        u[-2,:] = bd_xL(ys, ts[1])
    if bd_y0 is None:
        # Neumann
        u[:,0] = u[:,2]
    else:
        # Dirichlet
        u[:,1] = bd_y0(xs, ts[1])
    if bd_yL is None:
        # Neumann
        u[:,-1] = u[:,-3]
    else:
        # Dirichlet
        u[:,-2] = bd_yL(xs, ts[1])

    if callback is not None:
        callback(u[1:-1], xs, ys, ts, 1)

    # reference swap
    u, u_n, u_nm1 = u_nm1, u, u_n

    # compute each timestep until the end of the simulation
    for n in range(1, Nt):
        u[1:-1,1:-1] = 2 * u_n[1:-1,1:-1] - u_nm1[1:-1,1:-1] \
            + 0.5 * Cx2 * ((q[1:-1,1:-1] + q[2:,1:-1]) * (u_n[2:,1:-1] - u_n[1:-1,1:-1]) - (q[1:-1,1:-1] + q[:-2,1:-1]) * (u_n[1:-1,1:-1] - u_n[:-2,1:-1])) \
            + 0.5 * Cy2 * ((q[1:-1,1:-1] + q[1:-1,2:]) * (u_n[1:-1,2:] - u_n[1:-1,1:-1]) - (q[1:-1,1:-1] + q[1:-1,:-2]) * (u_n[1:-1,1:-1] - u_n[1:-1,:-2])) \
            + dt2 * f(xs[1:-1], ys[1:-1], ts[n])
        if bd_x0 is None:
            # Neumann
            u[0,:] = u[2,:]
        else:
            # Dirichlet
            u[1,:] = bd_x0(ys, ts[n])
        if bd_xL is None:
            # Neumann
            u[-1,:] = u[-3,:]
        else:
            # Dirichlet
            u[-2,:] = bd_xL(ys, ts[n])
        if bd_y0 is None:
            # Neumann
            u[:,0] = u[:,2]
        else:
            # Dirichlet
            u[:,1] = bd_y0(xs, ts[n])
        if bd_yL is None:
            # Neumann
            u[:,-1] = u[:,-3]
        else:
            # Dirichlet
            u[:,-2] = bd_yL(xs, ts[n])

        if callback is not None:
            if callback(u[1:-1], xs, ys, ts, n+1):
                break

        u, u_n, u_nm1 = u_nm1, u, u_n

    u = u_n
    cpu_time = time.process_time() - t0
    return u[1:-1], xs, ys, ts, cpu_time
