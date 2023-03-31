#include "../../wave2d.c"
#include "../../save_frame.c"

double small_gaussian(double x, double y, SimulationParameters *params);
bool elliptical_cavity(double x, double y, SimulationParameters *params);

SimulationParameters sim_params = {
  .initial = small_gaussian,
  .velocity = zero_velocity,
  .forcing = zero_forcing,
  .Lx = 5,
  .Ly = 5,
  .Nx = 2000,
  .Ny = 2000,
  .T = 10,
  .c = 1,
  .dt = -1,
  .Left_Border = DIRICHLET,
  .Right_Border = DIRICHLET,
  .Top_Border = DIRICHLET,
  .Bottom_Border = DIRICHLET,
  .obstacle = elliptical_cavity,
  .callback = save_frame_callback
};

PlottingParameters plot_params = {
  .plot_type = ENERGY,
  .umin = -4,
  .umax = 0,
  .image_scale = 1,
  .skip_frame = 10,
  .filename_prefix = "ellipse",
  .threading = true
};

int main() {
  double cpu = solver(sim_params);
  printf("cpu: %f\n", cpu);

  return 0;
}

void save_frame_callback(double *u, double *xs, double *ys, double t, int n, int Nt, SimulationParameters sim_params) {
  save_frame(u, xs, ys, t, n, Nt, sim_params, plot_params);
}

double small_gaussian(double x, double y, SimulationParameters *params) {
  double rad = 0.01;
  double cx = params->Lx * (2 - sqrt(3)) / 4.0; // at focal point
  double cy = params->Ly / 2.0;

  return 10 * exp((-(x - cx) * (x - cx) - (y - cy) * (y - cy)) / (2 * rad * rad));
}

bool elliptical_cavity(double x, double y, SimulationParameters *params) {
  double a = params->Lx / 2.0;
  double b = params->Ly / 4.0;
  return (x - params->Lx / 2.0) * (x - params->Lx / 2.0) / (a * a) + (y - params->Ly / 2.0) * (y - params->Ly / 2.0) / (b * b) > 1;
}
