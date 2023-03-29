#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

enum BoundaryType{NEUMANN, DIRICHLET};

typedef struct SimulationParameters {
  double (*initial)(double, double, struct SimulationParameters);
  double (*velocity)(double, double, struct SimulationParameters);
  double (*forcing)(double, double, double, struct SimulationParameters);
  double Lx;
  double Ly;
  size_t Nx;
  size_t Ny;
  double T;
  double c;
  double dt;
  enum BoundaryType Left_Border;
  enum BoundaryType Right_Border;
  enum BoundaryType Top_Border;
  enum BoundaryType Bottom_Border;
  bool (*obstacle)(double, double, struct SimulationParameters);
  void (*callback)(double*, double*, double*, double, int, int, struct SimulationParameters);
} SimulationParameters;

double solver(SimulationParameters params);
double plug(double x, double y, SimulationParameters params);
double zero_velocity(double x, double y, SimulationParameters params);
double zero_forcing(double x, double y, double t, SimulationParameters params);
bool zero_obstacle(double x, double y, SimulationParameters params);

int main2() {
  size_t width = 300;
  size_t height = 300;
  uint32_t *pixels = malloc(sizeof(uint32_t) * width * height);
  uint32_t COLOR_RED = 0xFF0000FF;
  uint32_t COLOR_BLUE = 0xFFFF0000;

  for (size_t j = 0; j < height; ++j) {
    for (size_t i = 0; i < width; ++i) {
      pixels[j * width + i] = COLOR_BLUE;
    }
  }

  //stbi_write_png("output.png", width, height, 4, pixels, width * 4);

  //double cpu = solver(1, -0.8);
  //printf("cpu: %f\n", cpu);

  free(pixels);

  return 0;
}

//solver for dirichlet boundary conditions
double solver(SimulationParameters params) {
  //initialize xs array
  double *xs = malloc(sizeof(double) * (params.Nx + 1));
  for (size_t i = 0; i <= params.Nx; ++i) {
    xs[i] = ((double) params.Lx * i) / params.Nx;
  }

  //initialize ys array
  double *ys = malloc(sizeof(double) * (params.Ny + 1));
  for (size_t j = 0; j <= params.Ny; ++j) {
    ys[j] = ((double) params.Ly * j) / params.Ny;
  }

  double stability_limit = (1 / params.c) / sqrt((params.Nx * params.Nx) / (params.Lx * params.Lx) + (params.Ny * params.Ny) / (params.Ly * params.Ly));
  if (params.dt <= 0) {
    double safety_factor = -params.dt;
    params.dt = safety_factor * stability_limit;
  }
  else if (params.dt > stability_limit) {
    printf("Warning: dt=%f exceeds the stability limit %f", params.dt, stability_limit);
  }

  int Nt = (int) params.T / params.dt; //total number of timesteps
  double dt2 = params.dt * params.dt;
  double Cx2 = dt2 * params.Nx * params.Nx / (params.Lx * params.Lx);
  double Cy2 = dt2 * params.Ny * params.Ny / (params.Ly * params.Ly);

  double c2 = params.c * params.c;
  double *q = malloc(sizeof(double) * (params.Nx + 3) * (params.Ny + 3)); //array filled with c(x,y)^2
  for (size_t index = 0; index < (params.Nx + 3) * (params.Ny + 3); ++index) {
    q[index] = c2;
  }

  double *u = malloc(sizeof(double) * (params.Nx + 3) * (params.Ny + 3));
  double *u_n = malloc(sizeof(double) * (params.Nx + 3) * (params.Ny + 3));
  double *u_nm1 = malloc(sizeof(double) * (params.Nx + 3) * (params.Ny + 3));
  double *temp; //used for reference swap later

  memset(u, 0, sizeof(double) * (params.Nx + 3) * (params.Ny + 3));
  memset(u_n, 0, sizeof(double) * (params.Nx + 3) * (params.Ny + 3));
  memset(u_nm1, 0, sizeof(double) * (params.Nx + 3) * (params.Ny + 3));

  clock_t t0 = clock();

  //initial condition
  for (size_t j = 1; j <= params.Ny + 1; ++j) {
    for (size_t i = 1; i <= params.Nx + 1; ++i) {
      size_t index = j * (params.Nx + 3) + i;
      if (params.obstacle(xs[i-1], ys[j-1], params)) {
        u_n[index] = 0;
      }
      else {
        u_n[index] = params.initial(xs[i-1], ys[j-1], params);
      }
    }
  }
  params.callback(u_n, xs, ys, 0, 0, Nt, params);

  //special first timestep formula
  for (size_t j = 1; j <= params.Ny + 1; ++j) {
    for (size_t i = 1; i <= params.Nx + 1; ++i) {
      size_t index = j * (params.Nx + 3) + i;

      //left border
      if (i == 1 && params.Left_Border == NEUMANN) {
        u[index] = 0;
      }
      //right border
      else if (i == params.Nx + 1 && params.Right_Border == NEUMANN) {
        u[index] = 0;
      }
      //top border
      else if (j == 1 && params.Top_Border == NEUMANN) {
        u[index] = 0;
      }
      //bottom border
      else if (j == params.Ny + 1 && params.Bottom_Border == NEUMANN) {
        u[index] = 0;
      }
      else if (params.obstacle(xs[i-1], xs[j-1], params)) {
        u[index] = 0;
      }
      else {
        u[index] = u_n[index] + params.dt * params.velocity(xs[i-1], ys[j-1], params)
          + 0.25 * Cx2 * ((q[index] + q[index + 1]) * (u_n[index + 1] - u_n[index]) - (q[index] + q[index - 1]) * (u_n[index] - u_n[index - 1]))
          + 0.25 * Cy2 * ((q[index] + q[index + params.Nx + 3]) * (u_n[index + params.Nx + 3] - u_n[index]) - (q[index] + q[index - params.Nx - 3]) * (u_n[index] - u_n[index - params.Nx - 3]))
          + 0.5 * dt2 * params.forcing(xs[i-1], ys[j-1], 0, params);
      }
    }
  }
  if (params.Left_Border == DIRICHLET) {
    for (size_t j = 1; j <= params.Ny + 1; ++j) {
      size_t outer_i = 0;
      size_t inner_i = 2;
      size_t outer_index = j * (params.Nx + 3) + outer_i;
      size_t inner_index = j * (params.Nx + 3) + inner_i;
      u[outer_index] = u[inner_index];
    }
  }
  if (params.Right_Border == DIRICHLET) {
    for (size_t j = 1; j <= params.Ny + 1; ++j) {
      size_t outer_i = params.Nx + 2;
      size_t inner_i = params.Nx;
      size_t outer_index = j * (params.Nx + 3) + outer_i;
      size_t inner_index = j * (params.Nx + 3) + inner_i;
      u[outer_index] = u[inner_index];
    }
  }
  if (params.Top_Border == DIRICHLET) {
    for (size_t i = 1; i <= params.Nx + 1; ++i) {
      size_t outer_j = 0;
      size_t inner_j = 2;
      size_t outer_index = outer_j * (params.Nx + 3) + i;
      size_t inner_index = inner_j * (params.Nx + 3) + i;
      u[outer_index] = u[inner_index];
    }
  }
  if (params.Bottom_Border == DIRICHLET) {
    for (size_t i = 1; i <= params.Nx + 1; ++i) {
      size_t outer_j = params.Ny + 2;
      size_t inner_j = params.Ny;
      size_t outer_index = outer_j * (params.Nx + 3) + i;
      size_t inner_index = inner_j * (params.Nx + 3) + i;
      u[outer_index] = u[inner_index];
    }
  }
  params.callback(u, xs, ys, params.dt, 1, Nt, params);

  //reference swap
  temp = u_nm1;
  u_nm1 = u_n;
  u_n = u;
  u = temp;

  //compute each timestep until the end of the simulation
  for (size_t n = 1; n < Nt; ++n) {
    double t = n * params.dt;
    for (size_t j = 1; j <= params.Ny + 1; ++j) {
      for (size_t i = 1; i <= params.Nx + 1; ++i) {
        size_t index = j * (params.Nx + 3) + i;
        //left border
        if (i == 1 && params.Left_Border == NEUMANN) {
          u[index] = 0;
        }
        //right border
        else if (i == params.Nx + 1 && params.Right_Border == NEUMANN) {
          u[index] = 0;
        }
        //top border
        else if (j == 1 && params.Top_Border == NEUMANN) {
          u[index] = 0;
        }
        //bottom border
        else if (j == params.Ny + 1 && params.Bottom_Border == NEUMANN) {
          u[index] = 0;
        }
        else if (params.obstacle(xs[i-1], xs[j-1], params)) {
          u[index] = 0;
        }
        else {
          u[index] = 2 * u_n[index] - u_nm1[index]
            + 0.5 * Cx2 * ((q[index] + q[index + 1]) * (u_n[index + 1] - u_n[index]) - (q[index] + q[index - 1]) * (u_n[index] - u_n[index - 1]))
            + 0.5 * Cy2 * ((q[index] + q[index + params.Nx + 3]) * (u_n[index + params.Nx + 3] - u_n[index]) - (q[index] + q[index - params.Nx - 3]) * (u_n[index] - u_n[index - params.Nx - 3]))
            + dt2 * params.forcing(xs[i-1], ys[j-1], t, params);
        }
      }
    }
    if (params.Left_Border == DIRICHLET) {
      for (size_t j = 1; j <= params.Ny + 1; ++j) {
        size_t outer_i = 0;
        size_t inner_i = 2;
        size_t outer_index = j * (params.Nx + 3) + outer_i;
        size_t inner_index = j * (params.Nx + 3) + inner_i;
        u[outer_index] = u[inner_index];
      }
    }
    if (params.Right_Border == DIRICHLET) {
      for (size_t j = 1; j <= params.Ny + 1; ++j) {
        size_t outer_i = params.Nx + 2;
        size_t inner_i = params.Nx;
        size_t outer_index = j * (params.Nx + 3) + outer_i;
        size_t inner_index = j * (params.Nx + 3) + inner_i;
        u[outer_index] = u[inner_index];
      }
    }
    if (params.Top_Border == DIRICHLET) {
      for (size_t i = 1; i <= params.Nx + 1; ++i) {
        size_t outer_j = 0;
        size_t inner_j = 2;
        size_t outer_index = outer_j * (params.Nx + 3) + i;
        size_t inner_index = inner_j * (params.Nx + 3) + i;
        u[outer_index] = u[inner_index];
      }
    }
    if (params.Bottom_Border == DIRICHLET) {
      for (size_t i = 1; i <= params.Nx + 1; ++i) {
        size_t outer_j = params.Ny + 2;
        size_t inner_j = params.Ny;
        size_t outer_index = outer_j * (params.Nx + 3) + i;
        size_t inner_index = inner_j * (params.Nx + 3) + i;
        u[outer_index] = u[inner_index];
      }
    }
    params.callback(u, xs, ys, t + params.dt, n + 1, Nt, params);

    //reference swap
    temp = u_nm1;
    u_nm1 = u_n;
    u_n = u;
    u = temp;
  }

  clock_t t1 = clock();
  double cpu = (double) (t1 - t0) / CLOCKS_PER_SEC;

  free(xs);
  free(ys);
  free(q);
  free(u);
  free(u_n);
  free(u_nm1);

  return cpu;
}

double plug(double x, double y, SimulationParameters params) {
  double rad = 1;
  if ((x - params.Lx / 2.0) * (x - params.Lx / 2.0) + (y - params.Ly / 2.0) * (y - params.Ly / 2.0) > rad * rad) {
    return 0;
  }
  else {
    return 1;
  }
}

double zero_velocity(double x, double y, SimulationParameters params) {
  return 0;
}

double zero_forcing(double x, double y, double t, SimulationParameters params) {
  return 0;
}

bool zero_obstacle(double x, double y, SimulationParameters params) {
  return false;
}
