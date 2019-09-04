#include <fstream>

#include <ilqr/ilqr.h>
#include <kinematics/rk4.h>

constexpr double delta_t = 1e-1;
constexpr double g = 9.80665;
constexpr double l = 1.0;
constexpr int timesteps = 20;
constexpr double friction_coeff = 0.0;
constexpr double max_torque = 100.0;

int main(int, char**)
{
  using ILQRController = controllers::ILQRController<2, 1>;

  auto calc_delta = [](const ILQRController::State_t& x,
                       const ILQRController::Control_t& u) -> ILQRController::State_t {
    ILQRController::State_t res;
    res << x(1), g / l * sin(x(0)) + u(0) - friction_coeff * x(1);
    return res;
  };

  auto dynamics = [=](const ILQRController::State_t& x, const ILQRController::Control_t& u) -> ILQRController::State_t {
    return kinematics::integrate(x, u, calc_delta, delta_t);
  };

  auto cost = [](const ILQRController::State_t& x, const ILQRController::Control_t& u) -> double {
    return 0 * x(0) * x(0) + 1e1 * u(0) * u(0);
  };

  auto final_cost = [](const ILQRController::State_t& x) -> double {
    std::cout << "x: " << x.transpose() << std::endl;
    return 1e6 * x(0) * x(0) + 1e4 * x(1) * x(1);
  };

  ILQRController controller(dynamics, cost, final_cost);

  ILQRController::State_t x;
  x << 2.5, 0.0;

  auto controls = controller.solve(x, timesteps);
  {
    std::ofstream csv("ilqr.csv");
    csv << "u,x0,x1\n";

    csv << "0," << x(0) << "," << x(1) << "\n";

    for (const auto& u : controls)
    {
      x = dynamics(x, u);
      std::cout << u(0) << "," << x(0) << "," << x(1) << "\n";
      csv << u(0) << "," << x(0) << "," << x(1) << "\n";
    }
  }

  return 0;
}
