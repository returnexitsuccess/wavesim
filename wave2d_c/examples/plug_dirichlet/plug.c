#define INITIAL plug
#define VELOCITY zero_velocity
#define FORCING zero_forcing

#define LX 5
#define LY 5
#define NX 1000
#define NY 1000
#define T 10

#define ALL_BORDERS DIRICHLET

#define CALLBACK save_frame
#define OBSTACLE zero_obstacle

#define PLOT_TYPE ENERGY
#define UMIN -4
#define UMAX 0

#define IMAGE_SCALE 1
#define SKIP_FRAME 10

#define FILENAME_PREFIX "plug"

#include "../../wave2d.c"

int main() {
  double cpu = solver(1, -0.8);
  printf("cpu: %f\n", cpu);

  return 0;
}
