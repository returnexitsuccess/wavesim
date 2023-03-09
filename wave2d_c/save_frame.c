#include "wave2d.h"

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
      exit(1);
    }
  }

  if (n % SKIP_FRAME != 0) {
    return;
  }

  uint32_t *pixels = malloc(sizeof(uint32_t) * (NX + 1) * (NY + 1) * IMAGE_SCALE * IMAGE_SCALE);

  for (size_t j = 1; j <= NY + 1; ++j) {
    for (size_t i = 1; i <= NX + 1; ++i) {
      size_t u_index = j * (NX + 3) + i;
      uint32_t color;
      if (OBSTACLE(xs[i-1], ys[j-1])) {
        color = 0xFF000000; //set obstacle to black
      }
      else {
        double u_norminv;
        if (PLOT_TYPE == LINEAR) {
          u_norminv = 4 * (u[u_index] - UMAX) / (UMIN - UMAX);
        }
        else if (PLOT_TYPE == ENERGY) {
          double u2 = u[u_index] * u[u_index];
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
        color = (0xFF000000) | (b << 16) | (g << 8) | (r);
      }

      for (size_t image_i = 0; image_i < IMAGE_SCALE; ++image_i) {
        for (size_t image_j = 0; image_j < IMAGE_SCALE; ++image_j) {
          size_t image_index = (IMAGE_SCALE * (j - 1) + image_j) * (IMAGE_SCALE * (NX + 1)) + (IMAGE_SCALE * (i - 1) + image_i);
          pixels[image_index] = color;
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

  free(pixels);
}