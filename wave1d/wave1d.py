#!/usr/bin/env python3

# Code adapted from
# Finite difference methods for wave equations
# by Langtangen and Linge

import time
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Solve wave equation Dt Dt u = Dx (c(x)^2 Dx u) + f
# in 1D on mesh grid with initial conditions I, V
# and Dirichlet or Neumann boundary conditions given by bd_0 and bd_L
def solver(I, V, f, c, bd_0, bd_L, L, T, dt, C, callback=None, stability_safety_factor=1):
    Nt = int(round(T / dt))
    ts = np.linspace(0, Nt * dt, Nt + 1)

    if isinstance(c, (float, int)):
        c_max = c
    elif callable(c):
        c_max = max([c(x) for x in np.linspace(0, L, 101)])
    dx = dt * c_max / (C * stability_safety_factor)
    Nx = int(round(L / dx))
    xs = np.linspace(0, L, Nx + 1)

    dt = ts[1] - ts[0]
    dx = xs[1] - xs[0]

    if isinstance(c, (float, int)):
        q = np.full(Nx + 1, c * c, dtype='float64')
    elif callable(c):
        cs = c(xs)
        q = cs * cs
    dt2 = dt * dt
    C2 = dt2 / (dx * dx)

    # allow ease of setting initial conditions or forcing term
    if f is None or f == 0:
        f = lambda x, t: 0
    if V is None or V == 0:
        V = lambda x: 0
    if bd_0 == 0:
        bd_0 = lambda t: 0
    if bd_L == 0:
        bd_L = lambda t: 0

    # initialize u arrays
    u = np.zeros(Nx + 1) # u array at new timestep
    u_n = np.zeros(Nx + 1) # u array at current timestep
    u_nm1 = np.zeros(Nx + 1) # u array at previous timestep

    t0 = time.process_time()

    # initial condition
    u_n = np.vectorize(I)(xs).astype('float64')

    if callback is not None:
        callback(u_n, xs, ts, 0)

    # special first timestep formula
    u[1:-1] = u_n[1:-1] + dt * V(xs[1:-1]) + 0.25 * C2 * ((q[1:-1] + q[2:]) * (u_n[2:] - u_n[1:-1]) - (q[1:-1] + q[:-2]) * (u_n[1:-1] - u_n[:-2])) + 0.5 * dt2 * f(xs[1:-1], 0)
    if bd_0 is None:
        # Neumann
        u[0] = u_n[0] + dt * V(xs[0]) + C2 * q[0] * (u_n[1] - u_n[0]) + 0.5 * dt2 * f(xs[0], 0)
    else:
        # Dirichlet
        u[0] = bd_0(ts[1])
    if bd_L is None:
        # Neumann
        u[-1] = u_n[-1] + dt * V(xs[-1]) + C2 * q[-1] * (u_n[-2] - u_n[-1]) + 0.5 * dt2 * f(xs[-1], 0)
    else:
        # Dirichlet
        u[-1] = bd_L(ts[1])

    if callback is not None:
        callback(u, xs, ts, 1)

    # reference swap
    u, u_n, u_nm1 = u_nm1, u, u_n

    # compute each timestep until the end of the simulation
    for n in range(1, Nt):
        u[1:-1] = 2 * u_n[1:-1] - u_nm1[1:-1] + 0.5 * C2 * ((q[1:-1] + q[2:]) * (u_n[2:] - u_n[1:-1]) - (q[1:-1] + q[:-2]) * (u_n[1:-1] - u_n[:-2])) + dt2 * f(xs[1:-1], ts[n])
        if bd_0 is None:
            # Neumann
            u[0] = 2 * u_n[0] - u_nm1[0] + 2 * C2 * q[0] * (u_n[1] - u_n[0]) + dt2 * f(xs[0], ts[n])
        else:
            #Dirichlet
            u[0] = bd_0(ts[n+1])
        if bd_L is None:
            # Neumann
            u[-1] = 2 * u_n[-1] - u_nm1[-1] + 2 * C2 * q[-1] * (u_n[-2] - u_n[-1]) + dt2 * f(xs[-1], ts[n])
        else:
            # Dirichlet
            u[-1] = bd_L(ts[n+1])

        if callback is not None:
            if callback(u, xs, ts, n+1):
                break

        u, u_n, u_nm1 = u_nm1, u, u_n

    u = u_n
    cpu_time = time.process_time() - t0
    return u, xs, ts, cpu_time


class PlotAndStoreSolution:
    def __init__(
            self,
            casename='tmp',
            umin=-1, umax=1,
            framerate=4,
            title='',
            skip_frame=1,
            filename=None):
        self.casename = casename
        self.yaxis = [umin, umax]
        self.framerate = framerate
        self.title = title
        self.skip_frame = skip_frame
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

    def __call__(self, u, x, t, n):
        # Save solution u to a file using numpy.savez
        if self.filename is not None:
            name = f"u{n:>04}"
            kwargs = {name: u}
            fname = '.' + self.filename + '_' + name + '.dat'
            np.savez(fname, **kwargs)
            self.t.append(t[n])
            if n == 0:
                np.savez('.' + self.filename + '_x.dat', x=x)

        # Animate
        if n % self.skip_frame != 0:
            return
        title = f"t={t[n]:.3f}"
        if self.title:
            title = self.title + ' ' + title
        if n == 0:
            self.plt.ion()
            self.lines = self.plt.plot(x, u, 'r-')
            self.plt.axis([x[0], x[-1], self.yaxis[0], self.yaxis[1]])
            self.plt.xlabel('x')
            self.plt.ylabel('u')
            self.plt.title(title)
            self.plt.legend([f"t={t[n]:.3f}"], loc='lower left')
        else:
            self.lines[0].set_ydata(u)
            self.plt.legend([f"t={t[n]:.3f}"], loc='lower left')
            self.plt.draw()

        self.plt.savefig(f"{self.casename}_{n:>04}.png")

    def saveVideo(self):
        filespec = f"{self.casename}_%04d.png"
        movie_program = 'ffmpeg'
        cmd = f"{movie_program} -hide_banner -loglevel warning -y -r {self.framerate} -i {filespec} -vcodec libx264 {self.casename}.mp4"
        os.system(cmd)



def demo_guitar(C):
    L = 0.75
    x0 = 0.8 * L
    a = 0.005
    freq = 440
    wavelength = 2 * L
    c = freq * wavelength
    omega = 2 * np.pi * freq
    num_periods = 1
    T = 2 * np.pi / omega * num_periods

    dt = L / (100 * c)

    I = lambda x: a * x / x0 if x < x0 else a * (L - x) / (L - x0)

    umin = -1.2 * a
    umax = -umin
    #cpu = viz(I, 0, 0, c, None, None, L, T, dt, C, umin, umax, solver_function=solver)

    plotter = PlotAndStoreSolution(umin=umin, umax=umax, casename='demo_guitar')
    u, x, t, cpu = solver(I, 0, 0, c, 0, 0, L, T, dt, C, callback=plotter)
    plotter.saveVideo()
    print(cpu)




def test_quadratic():
    """Check that u(x,t)=x(L-x)(1+t/2) is exactly reproduced."""

    def u_exact(x, t):
        return x*(L-x)*(1 + 0.5*t)

    def I(x):
        return u_exact(x, 0)

    def V(x):
        return 0.5*u_exact(x, 0)

    def f(x, t):
        return 2*(1 + 0.5*t)*c*c

    L = 2.5
    c = 1.5
    C = 0.75
    Nx = 6 # Very coarse mesh for this exact test
    dt = C*(L/Nx)/c
    T = 18

    def assert_no_error(u, x, t, n):
        u_e = u_exact(x, t[n])
        diff = np.abs(u - u_e).max()
        tol = 1E-13
        assert diff < tol

    solver(I, V, f, c, 0, 0, L, T, dt, C, callback=assert_no_error)

def test_constant():
    """
    Check the scalar and vectorized versions for
    a constant u(x,t). We simulate in [0, L] and apply
    Neumann and Dirichlet conditions at both ends.
    """
    u_const = 0.45
    u_exact = lambda x, t: u_const
    I = lambda x: u_exact(x, 0)
    V = lambda x: 0
    f = lambda x, t: 0

    def assert_no_error(u, x, t, n):
        u_e = u_exact(x, t[n])
        diff = np.abs(u - u_e).max()
        msg = f"diff={diff:E}, t_{n}={t[n]:g}"
        tol = 1E-13
        assert diff < tol, msg

    for U_0 in (None, lambda t: u_const):
        for U_L in (None, lambda t: u_const):
            L = 2.5
            c = 1.5
            C = 0.75
            Nx = 3 # Very coarse mesh for this exact test
            dt = C*(L/Nx)/c
            T = 18 # long time integration

            solver(I, V, f, c, U_0, U_L, L, T, dt, C, callback=assert_no_error)
            print(U_0, U_L)

def test_plug():
    """Check that an initial plug is correct back after one period."""
    L = 1.0
    c = 0.5
    dt = (L/10)/c # Nx=10

    I = lambda x: 0 if abs(x-L/2.0) > 0.1 else 1

    u, x, t, cpu = solver(I, 0, 0, c, None, None, L, 4, dt, 1)

    tol = 1E-13
    u_0 = np.array([I(x_) for x_ in x])
    diff = np.abs(u - u_0).max()
    assert diff < tol
