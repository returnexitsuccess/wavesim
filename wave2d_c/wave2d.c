#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "wave2d.h"

#include "save_frame.c"

// SIMULATION SETTINGS

#ifndef INITIAL
#define INITIAL plug
#endif

#ifndef VELOCITY
#define VELOCITY zero_velocity
#endif

#ifndef FORCING
#define FORCING zero_forcing
#endif

#ifndef LX
#define LX 5
#endif

#ifndef LY
#define LY 5
#endif

#ifndef NX
#define NX 100
#endif

#ifndef NY
#define NY 100
#endif

#ifndef T
#define T 10
#endif

#ifdef ALL_BORDERS
#define LEFT_BORDER ALL_BORDERS
#define RIGHT_BORDER ALL_BORDERS
#define TOP_BORDER ALL_BORDERS
#define BOTTOM_BORDER ALL_BORDERS
#endif

#ifndef LEFT_BORDER
#define LEFT_BORDER NEUMANN
#endif

#ifndef RIGHT_BORDER
#define RIGHT_BORDER NEUMANN
#endif

#ifndef TOP_BORDER
#define TOP_BORDER NEUMANN
#endif

#ifndef BOTTOM_BORDER
#define BOTTOM_BORDER NEUMANN
#endif

#ifndef CALLBACK
#define CALLBACK save_frame
#endif

#ifndef OBSTACLE
#define OBSTACLE zero_obstacle
#endif

// PLOTTING SETTINGS

#ifndef PLOT_TYPE
#define PLOT_TYPE ENERGY
#endif

#if PLOT_TYPE == LINEAR

#ifndef UMIN
#define UMIN -1
#endif

#ifndef UMAX
#define UMAX 1
#endif

#elif PLOT_TYPE == ENERGY

#ifndef UMIN
#define UMIN -4
#endif

#ifndef UMAX
#define UMAX 0
#endif

#endif

#ifndef IMAGE_SCALE
#define IMAGE_SCALE 10
#endif

#ifndef SKIP_FRAME
#define SKIP_FRAME 1
#endif

#ifndef FILENAME_PREFIX
#define FILENAME_PREFIX "output"
#endif


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

  double cpu = solver(1, -0.8);
  printf("cpu: %f\n", cpu);

  free(pixels);

  return 0;
}

//solver for dirichlet boundary conditions
double solver(double c, double dt) {
  //initialize xs array
  double *xs = malloc(sizeof(double) * (NX + 1));
  for (size_t i = 0; i <= NX; ++i) {
    xs[i] = ((double) LX * i) / NX;
  }

  //initialize ys array
  double *ys = malloc(sizeof(double) * (NY + 1));
  for (size_t j = 0; j <= NY; ++j) {
    ys[j] = ((double) LY * j) / NY;
  }

  double stability_limit = (1 / c) / sqrt((NX * NX) / (LX * LX) + (NY * NY) / (LY * LY));
  if (dt <= 0) {
    double safety_factor = -dt;
    dt = safety_factor * stability_limit;
  }
  else if (dt > stability_limit) {
    printf("Warning: dt=%f exceeds the stability limit %f", dt, stability_limit);
  }

  int Nt = (int) T / dt; //total number of timesteps
  double dt2 = dt * dt;
  double Cx2 = dt2 * NX * NX / (LX * LX);
  double Cy2 = dt2 * NY * NY / (LY * LY);

  double c2 = c * c;
  double *q = malloc(sizeof(double) * (NX + 3) * (NY + 3)); //array filled with c(x,y)^2
  for (size_t index = 0; index < (NX + 3) * (NY + 3); ++index) {
    q[index] = c2;
  }

  double *u = malloc(sizeof(double) * (NX + 3) * (NY + 3));
  double *u_n = malloc(sizeof(double) * (NX + 3) * (NY + 3));
  double *u_nm1 = malloc(sizeof(double) * (NX + 3) * (NY + 3));
  double *temp; //used for reference swap later

  memset(u, 0, sizeof(double) * (NX + 3) * (NY + 3));
  memset(u_n, 0, sizeof(double) * (NX + 3) * (NY + 3));
  memset(u_nm1, 0, sizeof(double) * (NX + 3) * (NY + 3));

  clock_t t0 = clock();

  //initial condition
  for (size_t j = 1; j <= NY + 1; ++j) {
    for (size_t i = 1; i <= NX + 1; ++i) {
      size_t index = j * (NX + 3) + i;
      if (OBSTACLE(xs[i-1], ys[j-1])) {
        u_n[index] = 0;
      }
      else {
        u_n[index] = INITIAL(xs[i-1], ys[j-1]);
      }
    }
  }
  CALLBACK(u_n, xs, ys, 0, 0);

  //special first timestep formula
  for (size_t j = 1; j <= NY + 1; ++j) {
    for (size_t i = 1; i <= NX + 1; ++i) {
      size_t index = j * (NX + 3) + i;

      //left border
      if (i == 1 && LEFT_BORDER == NEUMANN) {
        u[index] = 0;
      }
      //right border
      else if (i == NX + 1 && RIGHT_BORDER == NEUMANN) {
        u[index] = 0;
      }
      //top border
      else if (j == 1 && TOP_BORDER == NEUMANN) {
        u[index] = 0;
      }
      //bottom border
      else if (j == NY + 1 && BOTTOM_BORDER == NEUMANN) {
        u[index] = 0;
      }
      else if (OBSTACLE(xs[i-1], xs[j-1])) {
        u[index] = 0;
      }
      else {
        u[index] = u_n[index] + dt * VELOCITY(xs[i-1], ys[j-1])
          + 0.25 * Cx2 * ((q[index] + q[index + 1]) * (u_n[index + 1] - u_n[index]) - (q[index] + q[index - 1]) * (u_n[index] - u_n[index - 1]))
          + 0.25 * Cy2 * ((q[index] + q[index + NX + 3]) * (u_n[index + NX + 3] - u_n[index]) - (q[index] + q[index - NX - 3]) * (u_n[index] - u_n[index - NX - 3]))
          + 0.5 * dt2 * FORCING(xs[i-1], ys[j-1], 0);
      }
    }
  }
  if (LEFT_BORDER == DIRICHLET) {
    for (size_t j = 1; j <= NY + 1; ++j) {
      size_t outer_i = 0;
      size_t inner_i = 2;
      size_t outer_index = j * (NX + 3) + outer_i;
      size_t inner_index = j * (NX + 3) + inner_i;
      u[outer_index] = u[inner_index];
    }
  }
  if (RIGHT_BORDER == DIRICHLET) {
    for (size_t j = 1; j <= NY + 1; ++j) {
      size_t outer_i = NX + 2;
      size_t inner_i = NX;
      size_t outer_index = j * (NX + 3) + outer_i;
      size_t inner_index = j * (NX + 3) + inner_i;
      u[outer_index] = u[inner_index];
    }
  }
  if (TOP_BORDER == DIRICHLET) {
    for (size_t i = 1; i <= NX + 1; ++i) {
      size_t outer_j = 0;
      size_t inner_j = 2;
      size_t outer_index = outer_j * (NX + 3) + i;
      size_t inner_index = inner_j * (NX + 3) + i;
      u[outer_index] = u[inner_index];
    }
  }
  if (BOTTOM_BORDER == DIRICHLET) {
    for (size_t i = 1; i <= NX + 1; ++i) {
      size_t outer_j = NY + 2;
      size_t inner_j = NY;
      size_t outer_index = outer_j * (NX + 3) + i;
      size_t inner_index = inner_j * (NX + 3) + i;
      u[outer_index] = u[inner_index];
    }
  }
  CALLBACK(u, xs, ys, dt, 1);

  //reference swap
  temp = u_nm1;
  u_nm1 = u_n;
  u_n = u;
  u = temp;

  //compute each timestep until the end of the simulation
  for (size_t n = 1; n < Nt; ++n) {
    double t = n * dt;
    for (size_t j = 1; j <= NY + 1; ++j) {
      for (size_t i = 1; i <= NX + 1; ++i) {
        size_t index = j * (NX + 3) + i;
        //left border
        if (i == 1 && LEFT_BORDER == NEUMANN) {
          u[index] = 0;
        }
        //right border
        else if (i == NX + 1 && RIGHT_BORDER == NEUMANN) {
          u[index] = 0;
        }
        //top border
        else if (j == 1 && TOP_BORDER == NEUMANN) {
          u[index] = 0;
        }
        //bottom border
        else if (j == NY + 1 && BOTTOM_BORDER == NEUMANN) {
          u[index] = 0;
        }
        else if (OBSTACLE(xs[i-1], xs[j-1])) {
          u[index] = 0;
        }
        else {
          u[index] = 2 * u_n[index] - u_nm1[index]
            + 0.5 * Cx2 * ((q[index] + q[index + 1]) * (u_n[index + 1] - u_n[index]) - (q[index] + q[index - 1]) * (u_n[index] - u_n[index - 1]))
            + 0.5 * Cy2 * ((q[index] + q[index + NX + 3]) * (u_n[index + NX + 3] - u_n[index]) - (q[index] + q[index - NX - 3]) * (u_n[index] - u_n[index - NX - 3]))
            + dt2 * FORCING(xs[i-1], ys[j-1], t);
        }
      }
    }
    if (LEFT_BORDER == DIRICHLET) {
      for (size_t j = 1; j <= NY + 1; ++j) {
        size_t outer_i = 0;
        size_t inner_i = 2;
        size_t outer_index = j * (NX + 3) + outer_i;
        size_t inner_index = j * (NX + 3) + inner_i;
        u[outer_index] = u[inner_index];
      }
    }
    if (RIGHT_BORDER == DIRICHLET) {
      for (size_t j = 1; j <= NY + 1; ++j) {
        size_t outer_i = NX + 2;
        size_t inner_i = NX;
        size_t outer_index = j * (NX + 3) + outer_i;
        size_t inner_index = j * (NX + 3) + inner_i;
        u[outer_index] = u[inner_index];
      }
    }
    if (TOP_BORDER == DIRICHLET) {
      for (size_t i = 1; i <= NX + 1; ++i) {
        size_t outer_j = 0;
        size_t inner_j = 2;
        size_t outer_index = outer_j * (NX + 3) + i;
        size_t inner_index = inner_j * (NX + 3) + i;
        u[outer_index] = u[inner_index];
      }
    }
    if (BOTTOM_BORDER == DIRICHLET) {
      for (size_t i = 1; i <= NX + 1; ++i) {
        size_t outer_j = NY + 2;
        size_t inner_j = NY;
        size_t outer_index = outer_j * (NX + 3) + i;
        size_t inner_index = inner_j * (NX + 3) + i;
        u[outer_index] = u[inner_index];
      }
    }
    CALLBACK(u, xs, ys, t + dt, n + 1);

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

double plug(double x, double y) {
  double rad = 1;
  if ((x - LX / 2.0) * (x - LX / 2.0) + (y - LY / 2.0) * (y - LY / 2.0) > rad * rad) {
    return 0;
  }
  else {
    return 1;
  }
}

double zero_velocity(double x, double y) {
  return 0;
}

double zero_forcing(double x, double y, double t) {
  return 0;
}

bool zero_obstacle(double x, double y) {
  return false;
}
