#include "../../wave2d.c"
#include "../../save_frame.c"

SimulationParameters sim_params = {
  .initial = plug,
  .velocity = zero_velocity,
  .forcing = zero_forcing,
  .Lx = 5,
  .Ly = 5,
  .Nx = 1000,
  .Ny = 1000,
  .T = 10,
  .c = 1,
  .dt = -0.8,
  .Left_Border = DIRICHLET,
  .Right_Border = DIRICHLET,
  .Top_Border = DIRICHLET,
  .Bottom_Border = DIRICHLET,
  .obstacle = zero_obstacle,
  .callback = save_frame_callback
};

PlottingParameters plot_params = {
  .plot_type = ENERGY,
  .umin = -4,
  .umax = 0,
  .image_scale = 1,
  .skip_frame = 10,
  .filename_prefix = "plug",
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
