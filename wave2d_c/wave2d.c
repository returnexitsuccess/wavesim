#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb_image_write.h"

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

#ifndef CALLBACK
#define CALLBACK save_frame
#endif

#ifndef OBSTACLE
#define OBSTACLE zero_obstacle
#endif

// PLOTTING SETTINGS

#define LINEAR 0
#define ENERGY 1

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


double solver(double c, double dt);
void save_frame(double *u, double *xs, double *ys, double t, int n);
double plug(double x, double y);
double zero_velocity(double x, double y);
double zero_forcing(double x, double y, double t);
bool zero_obstacle(double x, double y);

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
  double *q = malloc(sizeof(double) * (NX + 1) * (NY + 1)); //array filled with c(x,y)^2
  for (size_t index = 0; index < (NX + 1) * (NY + 1); ++index) {
    q[index] = c2;
  }

  double *u = malloc(sizeof(double) * (NX + 1) * (NY + 1));
  double *u_n = malloc(sizeof(double) * (NX + 1) * (NY + 1));
  double *u_nm1 = malloc(sizeof(double) * (NX + 1) * (NY + 1));
  double *temp; //used for reference swap later

  clock_t t0 = clock();

  //initial condition
  for (size_t j = 0; j <= NY; ++j) {
    for (size_t i = 0; i <= NX; ++i) {
      size_t index = j * (NX + 1) + i;
      if (OBSTACLE(xs[i], ys[j])) {
        u_n[index] = 0;
      }
      else {
        u_n[index] = INITIAL(xs[i], ys[j]);
      }
    }
  }
  CALLBACK(u_n, xs, ys, 0, 0);

  //special first timestep formula
  for (size_t j = 0; j <= NY; ++j) {
    for (size_t i = 0; i <= NX; ++i) {
      size_t index = j * (NX + 1) + i;

      //left border
      if (i == 0) {
        u[index] = 0;
      }
      //right border
      else if (i == NX) {
        u[index] = 0;
      }
      //top border
      else if (j == 0) {
        u[index] = 0;
      }
      //bottom border
      else if (j == NX) {
        u[index] = 0;
      }
      else if (OBSTACLE(xs[i], xs[j])) {
        u[index] = 0;
      }
      else {
        u[index] = u_n[index] + dt * VELOCITY(xs[i], ys[j])
          + 0.25 * Cx2 * ((q[index] + q[index + 1]) * (u_n[index + 1] - u_n[index]) - (q[index] + q[index - 1]) * (u_n[index] - u_n[index - 1]))
          + 0.25 * Cy2 * ((q[index] + q[index + NX + 1]) * (u_n[index + NX + 1] - u_n[index]) - (q[index] + q[index - NX - 1]) * (u_n[index] - u_n[index - NX - 1]))
          + 0.5 * dt2 * FORCING(xs[i], ys[j], 0);
      }
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
    for (size_t j = 0; j <= NY; ++j) {
      for (size_t i = 0; i <= NX; ++i) {
        size_t index = j * (NX + 1) + i;
        //left border
        if (i == 0) {
          u[index] = 0;
        }
        //right border
        else if (i == NX) {
          u[index] = 0;
        }
        //top border
        else if (j == 0) {
          u[index] = 0;
        }
        //bottom border
        else if (j == NX) {
          u[index] = 0;
        }
        else if (OBSTACLE(xs[i], ys[j])) {
          u[index] = 0;
        }
        else {
          u[index] = 2 * u_n[index] - u_nm1[index]
            + 0.5 * Cx2 * ((q[index] + q[index + 1]) * (u_n[index + 1] - u_n[index]) - (q[index] + q[index - 1]) * (u_n[index] - u_n[index - 1]))
            + 0.5 * Cy2 * ((q[index] + q[index + NX + 1]) * (u_n[index + NX + 1] - u_n[index]) - (q[index] + q[index - NX - 1]) * (u_n[index] - u_n[index - NX - 1]))
            + dt2 * FORCING(xs[i], ys[j], t);
        }
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
  return cpu;
}

void save_frame(double *u, double *xs, double *ys, double t, int n) {
  if (n == 0) {
    //setup
    char command[256];
    size_t len = sizeof(command);

    int written = snprintf(command, len, "find . -maxdepth 1 -name '%s_*.png' -type f -delete", FILENAME_PREFIX);
    if (written < 0 || written > len) {
      fprintf(stderr, "Failed to create command with FILENAME_PREFIX %s: %s\n", FILENAME_PREFIX, strerror(errno));
      exit(1);
    }

    if (system(command) < 0) { //delete all .png files matching the FILENAME_PREFIX
      fprintf(stderr, "Failed to execute system call: %s\n", strerror(errno));
    }
  }

  if (n % SKIP_FRAME != 0) {
    return;
  }

  uint32_t *pixels = malloc(sizeof(uint32_t) * (NX + 1) * (NY + 1) * IMAGE_SCALE * IMAGE_SCALE);

  for (size_t j = 0; j <= NY; ++j) {
    for (size_t i = 0; i <= NX; ++i) {
      size_t index = j * (NX + 1) + i;
      if (OBSTACLE(xs[i], ys[j])) {
        pixels[index] = 0xFF000000; //set obstacle to black
      }
      else {
        double u_norminv;
        if (PLOT_TYPE == LINEAR) {
          u_norminv = 4 * (u[index] - UMAX) / (UMIN - UMAX);
        }
        else if (PLOT_TYPE == ENERGY) {
          double u2 = u[index] * u[index];
          if (u2 < pow(10, UMIN)) {
            u_norminv = 4;
          }
          else if (u2 > pow(10, UMAX)) {
            u_norminv = 0;
          }
          else {
            u_norminv = 4 * (log10(u2) - UMAX) / (UMIN - UMAX);
          }
        }
        uint8_t u_int = (uint8_t) u_norminv;
        uint8_t u_frac = (uint8_t) (256 * (u_norminv - u_int));
        uint32_t r, g, b;
        switch(u_int) {
        case 0 :
          r = 255;
          g = u_frac;
          b = 0;
          break;
        case 1 :
          r = 255 - u_frac;
          g = 255;
          b = 0;
          break;
        case 2 :
          r = 0;
          g = 255;
          b = u_frac;
          break;
        case 3 :
          r = 0;
          g = 255 - u_frac;
          b = 255;
          break;
        case 4 :
          r = 0;
          g = 0;
          b = 255;
          break;
        }
        uint32_t color = (0xFF000000) | (b << 16) | (g << 8) | (r);
        for (size_t image_i = 0; image_i < IMAGE_SCALE; ++image_i) {
          for (size_t image_j = 0; image_j < IMAGE_SCALE; ++image_j) {
            size_t image_index = (IMAGE_SCALE * j + image_j) * (IMAGE_SCALE * (NX + 1)) + (IMAGE_SCALE * i + image_i);
            pixels[image_index] = color;
          }
        }
      }
    }
  }

  char filename[100];
  int written = snprintf(filename, sizeof(filename), "%s_%04d.png", FILENAME_PREFIX, n);
  if (written < 0 || written > sizeof(filename)) {
    fprintf(stderr, "Failed to create filename %s_%04d.png: %s\n", FILENAME_PREFIX, n, strerror(errno));
    exit(1);
  }

  stbi_write_png(filename, IMAGE_SCALE * (NX + 1), IMAGE_SCALE * (NY + 1), 4, pixels, IMAGE_SCALE * (NX + 1) * 4);
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
