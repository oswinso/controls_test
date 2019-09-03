#include <ilqr/ilqr.h>

constexpr double delta_t = 1e-2;
constexpr double g = 9.80665;
constexpr double l = 1.0;
constexpr int timesteps = 1000;
constexpr double friction_coeff = 0.1;
constexpr double max_torque = 100.0;

int main(int argc, char** argv)
{
  using ILQRController = controllers::ILQRController<2, 1>;
}
