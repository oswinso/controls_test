#include <fstream>

#include <ilqr/ilqr.h>
#include <kinematics/rk4.h>

constexpr double delta_t = 1e-1;
constexpr double g = 9.80665;
constexpr double l = 1.0;
constexpr int timesteps = 50;
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
//    return 1e1 * x(0) * x(0) + 1e-9 * (u(0) * u(0));
    constexpr double alpha = 0.5;
    constexpr double beta = 0.9;
    double u_cost = alpha*alpha*(std::cosh(beta*u(0)/alpha)-1);
//    double angle_diff = std::abs(std::atan2(std::sin(x(0)), std::cos(x(0))));
//    double x_cost = 1e7 * angle_diff * angle_diff;
    double x_cost = 1e3 * x(0) * x(0);
//    printf("u: %8e, cost: %8e\n", u(0), cost);
    return u_cost + x_cost;
  };

  auto final_cost = [](const ILQRController::State_t& x) -> double {
//    std::cout << "x: " << x.transpose() << std::endl;
    return 1e4 * (x(0) * x(0)) + 1e2 * (x(1) * x(1));
  };

  ILQRController controller(dynamics, cost, final_cost);

  ILQRController::State_t x;
  x << 2.5, 0.0;

  auto controls = controller.solve(x, timesteps);
  std::cout << "\n\n>> DONE! <<\n\n";
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
