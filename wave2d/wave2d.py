#!/usr/bin/env python3

# Code adapted from
# Finite difference methods for wave equations
# by Langtangen and Linge

import time
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor, as_completed

# Solve wave equation Dt Dt u = D (c(x,y)^2 D u) + f
# in 2D rectangular domain on mesh grid with initial conditions I, V
# and Dirichlet or Neumann boundary conditions
def solver(I, V, f, c, bds, Lx, Ly, T, dt, Nx, Ny, callback=None):
    xs = np.linspace(0, Lx, Nx + 1).reshape((Nx + 1, 1))
    ys = np.linspace(0, Ly, Ny + 1).reshape((1, Ny + 1))

    dx = xs[1,0] - xs[0,0]
    dy = ys[0,1] - ys[0,0]


    stability_limit = (1 / float(c)) / np.sqrt(1 / (dx * dx) + 1 / (dy * dy))
    if dt <= 0:
        safety_factor = -dt
        dt = safety_factor * stability_limit
    elif dt > stability_limit:
        print(f"Warning: dt={dt} exceeds the stability limit {stability_limit}")

    Nt = int(round(T / dt))
    ts = np.linspace(0, Nt * dt, Nt + 1)

    if isinstance(c, (float, int)):
        q = np.full((Nx + 3, Ny + 3), c * c, dtype='float64')
    elif callable(c):
        cs = np.vectorize(c)(xs, ys).astype('float64')
        q = np.zeros((Nx + 3, Ny + 3))
        q[1:-1,1:-1] = cs * cs
        q[0,:] = q[2,:]
        q[-1,:] = q[-3,:]
        q[:,0] = q[:,2]
        q[:,-1] = q[:,-3]
    dt2 = dt * dt
    Cx2 = dt2 / (dx * dx)
    Cy2 = dt2 / (dy * dy)

    bd_x0, bd_xL, bd_y0, bd_yL = bds

    # allow ease of setting initial conditions or forcing term
    if f is None or f == 0:
        f = lambda x, y, t: 0
    if V is None or V == 0:
        V = lambda x, y: 0
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
        callback(u_n[1:-1,1:-1], xs, ys, ts, 0)

    # special first timestep formula
    u[1:-1,1:-1] = u_n[1:-1,1:-1] \
        + dt * V(xs, ys) \
        + 0.25 * Cx2 * ((q[1:-1,1:-1] + q[2:,1:-1]) * (u_n[2:,1:-1] - u_n[1:-1,1:-1]) - (q[1:-1,1:-1] + q[:-2,1:-1]) * (u_n[1:-1,1:-1] - u_n[:-2,1:-1])) \
        + 0.25 * Cy2 * ((q[1:-1,1:-1] + q[1:-1,2:]) * (u_n[1:-1,2:] - u_n[1:-1,1:-1]) - (q[1:-1,1:-1] + q[1:-1,:-2]) * (u_n[1:-1,1:-1] - u_n[1:-1,:-2])) \
        + 0.5 * dt2 * f(xs, ys, 0)
    if bd_x0 is None:
        # Neumann
        u[0,:] = u[2,:]
    else:
        # Dirichlet
        u[1,1:-1] = bd_x0(ys, ts[1])[:,0]
    if bd_xL is None:
        # Neumann
        u[-1,:] = u[-3,:]
    else:
        # Dirichlet
        u[-2,1:-1] = bd_xL(ys, ts[1])[:,0]
    if bd_y0 is None:
        # Neumann
        u[:,0] = u[:,2]
    else:
        # Dirichlet
        u[1:-1,1] = bd_y0(xs, ts[1])[0,:]
    if bd_yL is None:
        # Neumann
        u[:,-1] = u[:,-3]
    else:
        # Dirichlet
        u[1:-1,-2] = bd_yL(xs, ts[1])[0,:]

    if callback is not None:
        callback(u[1:-1,1:-1], xs, ys, ts, 1)

    # reference swap
    u, u_n, u_nm1 = u_nm1, u, u_n

    # compute each timestep until the end of the simulation
    for n in range(1, Nt):
        u[1:-1,1:-1] = 2 * u_n[1:-1,1:-1] - u_nm1[1:-1,1:-1] \
            + 0.5 * Cx2 * ((q[1:-1,1:-1] + q[2:,1:-1]) * (u_n[2:,1:-1] - u_n[1:-1,1:-1]) - (q[1:-1,1:-1] + q[:-2,1:-1]) * (u_n[1:-1,1:-1] - u_n[:-2,1:-1])) \
            + 0.5 * Cy2 * ((q[1:-1,1:-1] + q[1:-1,2:]) * (u_n[1:-1,2:] - u_n[1:-1,1:-1]) - (q[1:-1,1:-1] + q[1:-1,:-2]) * (u_n[1:-1,1:-1] - u_n[1:-1,:-2])) \
            + dt2 * f(xs, ys, ts[n])
        if bd_x0 is None:
            # Neumann
            u[0,:] = u[2,:]
        else:
            # Dirichlet
            u[1,1:-1] = bd_x0(ys, ts[n])[:,0]
        if bd_xL is None:
            # Neumann
            u[-1,:] = u[-3,:]
        else:
            # Dirichlet
            u[-2,1:-1] = bd_xL(ys, ts[n])[:,0]
        if bd_y0 is None:
            # Neumann
            u[:,0] = u[:,2]
        else:
            # Dirichlet
            u[1:-1,1] = bd_y0(xs, ts[n])[0,:]
        if bd_yL is None:
            # Neumann
            u[:,-1] = u[:,-3]
        else:
            # Dirichlet
            u[1:-1,-2] = bd_yL(xs, ts[n])[0,:]

        if callback is not None:
            if callback(u[1:-1,1:-1], xs, ys, ts, n+1):
                break

        u, u_n, u_nm1 = u_nm1, u, u_n

    u = u_n
    cpu_time = time.process_time() - t0
    return u[1:-1], xs, ys, ts, cpu_time


class PlotAndStoreSolution:
    def __init__(
            self,
            casename='tmp',
            umin=-1, umax=1,
            framerate=4,
            scale=1,
            title='',
            skip_frame=1,
            threads=1,
            filename=None):
        self.casename = casename
        self.yaxis = [umin, umax]
        self.framerate = framerate
        self.scale = scale
        self.title = title
        self.skip_frame = skip_frame
        self.threads = threads
        self.filename = filename

        import matplotlib.pyplot as plt
        self.plt = plt

        if filename is not None:
            self.t = []
            filenames = glob.glob('.' + self.filename + '*.dat.npz')
            for f in filenames:
                os.remove(f)

        for f in glob.glob(self.casename + '_*.png'):
            os.remove(f)

        if self.threads > 1:
            # optimal thread count may differ based on cpu
            self.ex = ThreadPoolExecutor(4)
            self.futures = []

    def __call__(self, u, x, y, t, n):
        # Save solution u to a file using numpy.savez
        if self.filename is not None:
            name = f"u{n:>04}"
            kwargs = {name: u}
            fname = '.' + self.filename + '_' + name + '.dat'
            np.savez(fname, **kwargs)
            self.t.append(t[n])
            if n == 0:
                np.savez('.' + self.filename + '_xy.dat', x=x, y=y)

        # Animate
        if n % self.skip_frame != 0:
            return

        if self.scale > 1:
            # scale image for saving
            u = np.kron(u, np.ones((self.scale, self.scale)))

        if self.threads > 1:
            f = self.ex.submit(self.plt.imsave, f"{self.casename}_{n:>04}.png", u, cmap='jet', vmin=self.yaxis[0], vmax=self.yaxis[1], origin='lower')
            self.futures.append(f)
        else:
            self.plt.imsave(f"{self.casename}_{n:>04}.png", u, cmap='jet', vmin=self.yaxis[0], vmax=self.yaxis[1], origin='lower')


        if n == len(t) - 1:
            self.plt.close()

    def saveVideo(self):
        if self.threads > 1:
            # make sure all threads completed
            for f in as_completed(self.futures):
                pass

        filespec = f"{self.casename}_*.png"
        movie_program = 'ffmpeg'
        cmd = f"{movie_program} -hide_banner -loglevel warning -y -r {self.framerate} -pattern_type glob -i \"{filespec}\" -vcodec libx264 {self.casename}.mp4"
        os.system(cmd)


def plug():
    Lx = 5
    Ly = 5
    rad2 = 1

    I = lambda x, y: 0 if (x - Lx / 2)**2 + (y - Ly / 2)**2 > rad2 else 1

    plotter = PlotAndStoreSolution(scale=5, skip_frame=1, framerate=30, threads=2)
    u, x, y, t, cpu = solver(I=I, V=0, f=0, c=1, bds=[None, None, None, None], Lx=Lx, Ly=Ly, T=20, dt=0.01, Nx=100, Ny=100, callback=plotter)
    plotter.saveVideo()
    return cpu

# print(f"cpu: {plug()}")


########## Tests ##########

def test_quadratic():
    """Check that u(x, y, t) = x(Lx - x)y(Ly - y)(1 + t/2) is exactly reproduced."""

    Lx = 2.5
    Ly = 2.5
    c = 1.5
    Nx = 6
    Ny = 6
    T = 18

    u_exact = lambda x, y, t: x * (Lx - x) * y * (Ly - y) * (1 + 0.5 * t)

    I = lambda x, y: u_exact(x, y, 0)

    V = lambda x, y: 0.5 * u_exact(x, y, 0)

    f = lambda x, y, t: 2 * c * c * (1 + 0.5 * t) * (x * (Lx - x) + y * (Ly - y))

    def assert_no_error(u, x, y, t, n):
        u_e = u_exact(x, y, t[n])
        diff = np.abs(u - u_e).max()
        tol = 1E-13
        assert diff < tol

    solver(I=I, V=V, f=f, c=c, bds=[0, 0, 0, 0], Lx=Lx, Ly=Ly, T=T, dt=-1, Nx=Nx, Ny=Ny, callback=assert_no_error)

def test_plotter():
    """Test the plotting class for different values of scale, skip_frame, and threads."""
    I = lambda x, y: 0 if (x-2.5)**2 + (y-2.5)**2 > 1 else 1

    for scale in (1, 5):
        for skip_frame in (1, 5):
            for threads in (1, 4):
                plotter = PlotAndStoreSolution(scale=scale, skip_frame=skip_frame, threads=threads, framerate=30)
                solver(I=I, V=0, f=0, c=1, bds=[None, None, None, None], Lx=5, Ly=5, T=1, dt=0.01, Nx=100, Ny=100, callback=plotter)
                plotter.saveVideo()
