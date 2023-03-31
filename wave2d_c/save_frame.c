#pragma once

#include <stdbool.h>
#include <pthread.h>

#include "wave2d.c"

enum PlotType{ENERGY, LINEAR};

typedef struct {
  enum PlotType plot_type;
  double umin;
  double umax;
  size_t image_scale;
  size_t skip_frame;
  char* filename_prefix;
  bool threading;
} PlottingParameters;

void save_frame(double *u, double *xs, double *ys, double t, int n, int Nt, SimulationParameters sim_params, PlottingParameters plot_params);
void *threading_helper(void *arguments);
void save_frame_callback(double *u, double *xs, double *ys, double t, int n, int Nt, SimulationParameters sim_params);

struct arg_struct {
  char *filename;
  int x;
  int y;
  int comp;
  void *data;
  int stride_bytes;
};

uint32_t **all_pixels;
char **filenames;
pthread_t *threads;
struct arg_struct *arg_list;

void save_frame(double *u, double *xs, double *ys, double t, int n, int Nt, SimulationParameters sim_params, PlottingParameters plot_params) {
  if (n == 0) {
    //setup
    char command[256];
    size_t len = sizeof(command);

    int written = snprintf(command, len, "find . -maxdepth 1 -name '%s_*.png' -type f -delete", plot_params.filename_prefix);
    if (written < 0 || written > len) {
      fprintf(stderr, "Failed to create command with FILENAME_PREFIX %s: %s\n", plot_params.filename_prefix, strerror(errno));
      exit(1);
    }

    if (system(command) < 0) { //delete all .png files matching the FILENAME_PREFIX
      fprintf(stderr, "Failed to execute system call: %s\n", strerror(errno));
      exit(1);
    }

    size_t frames = (Nt / plot_params.skip_frame) + 1;
    all_pixels = malloc(sizeof(uint32_t*) * frames);
    filenames = malloc(sizeof(char*) * frames);
    threads = malloc(sizeof(pthread_t) * frames);
    arg_list = malloc(sizeof(struct arg_struct) * frames);
  }

  if (n % plot_params.skip_frame == 0) {
    //uint32_t *pixels = malloc(sizeof(uint32_t) * (sim_params.Nx + 1) * (sim_params.Ny + 1) * plot_params.image_scale * plot_params.image_scale);
    all_pixels[n / plot_params.skip_frame] = malloc(sizeof(uint32_t) * (sim_params.Nx + 1) * (sim_params.Ny + 1) * plot_params.image_scale * plot_params.image_scale);
    uint32_t *pixels = all_pixels[n / plot_params.skip_frame];

    for (size_t j = 1; j <= sim_params.Ny + 1; ++j) {
      for (size_t i = 1; i <= sim_params.Nx + 1; ++i) {
        size_t u_index = j * (sim_params.Nx + 3) + i;
        uint32_t color;
        if (sim_params.obstacle(xs[i-1], ys[j-1], &sim_params)) {
          color = 0xFF000000; //set obstacle to black
        }
        else {
          double u_norminv;
          if (plot_params.plot_type == LINEAR) {
            u_norminv = 4 * (u[u_index] - plot_params.umax) / (plot_params.umin - plot_params.umax);
          }
          else if (plot_params.plot_type == ENERGY) {
            double u2 = u[u_index] * u[u_index];
            if (u2 < pow(10, plot_params.umin)) {
              u_norminv = 4;
            }
            else if (u2 > pow(10, plot_params.umax)) {
              u_norminv = 0;
            }
            else {
              u_norminv = 4 * (log10(u2) - plot_params.umax) / (plot_params.umin - plot_params.umax);
            }
          } else {
            fprintf(stderr, "Value of plot_type parameter not recognized");
            exit(1);
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
          default :
            fprintf(stderr, "u_int had unexpected value %d", u_int);
            exit(1);
          }
          color = (0xFF000000) | (b << 16) | (g << 8) | (r);
        }

        for (size_t image_i = 0; image_i < plot_params.image_scale; ++image_i) {
          for (size_t image_j = 0; image_j < plot_params.image_scale; ++image_j) {
            size_t image_index = (plot_params.image_scale * (j - 1) + image_j) * (plot_params.image_scale * (sim_params.Nx + 1)) + (plot_params.image_scale * (i - 1) + image_i);
            pixels[image_index] = color;
          }
        }
      }
    }
    
    size_t buffer_size = 100;
    filenames[n / plot_params.skip_frame] = malloc(sizeof(char) * buffer_size);
    char *filename = filenames[n / plot_params.skip_frame];
    int written = snprintf(filename, buffer_size, "%s_%04d.png", plot_params.filename_prefix, n);
    if (written < 0 || written > buffer_size) {
      fprintf(stderr, "Failed to create filename %s_%04d.png: %s\n", plot_params.filename_prefix, n, strerror(errno));
      exit(1);
    }

    if (plot_params.threading) {
      arg_list[n / plot_params.skip_frame] = (struct arg_struct) {
        .filename = filename,
        .x = plot_params.image_scale * (sim_params.Nx + 1),
        .y = plot_params.image_scale * (sim_params.Ny + 1),
        .comp = 4,
        .data = pixels,
        .stride_bytes = plot_params.image_scale * (sim_params.Nx + 1) * 4
      };
      if (pthread_create(&(threads[n / plot_params.skip_frame]), NULL, threading_helper, (void *)&(arg_list[n / plot_params.skip_frame])) != 0) {
        fprintf(stderr, "Failed to create thread: %s\n", strerror(errno));
        exit(1);
      }
    } else {
      stbi_write_png(filename, plot_params.image_scale * (sim_params.Nx + 1), plot_params.image_scale * (sim_params.Ny + 1), 4, pixels, plot_params.image_scale * (sim_params.Nx + 1) * 4);
    }
  }

  //cleanup
  if (n == Nt) {
    for (size_t frame = 0; frame <= (Nt / plot_params.skip_frame); ++frame) {
      pthread_join(threads[frame], NULL);
      //free(all_pixels[frame]);
      //free(filenames[frame]);
    }
    free(all_pixels);
    free(filenames);
    free(threads);
    free(arg_list);
  }
}

void *threading_helper(void *arguments) {
  struct arg_struct *args = arguments;

  stbi_write_png(args->filename, args->x, args->y, args->comp, args->data, args->stride_bytes);
  
  free(args->filename);
  free(args->data);

  return NULL;
}