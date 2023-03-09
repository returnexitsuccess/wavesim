#include <stdbool.h>

#define NEUMANN 0
#define DIRICHLET 1

#define LINEAR 0
#define ENERGY 1

double solver(double c, double dt);
void save_frame(double *u, double *xs, double *ys, double t, int n);
double plug(double x, double y);
double zero_velocity(double x, double y);
double zero_forcing(double x, double y, double t);
bool zero_obstacle(double x, double y);