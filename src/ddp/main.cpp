#include <fstream>

#include <ddp/ddp.h>
#include <kinematics/rk4.h>
#include <utils/utils.h>

constexpr double delta_t = 1e-1;
constexpr double g = 9.80665;
constexpr double l = 1.0;
constexpr int timesteps = 50;
constexpr double friction_coeff = 0.0;
constexpr double max_torque = 100.0;

int main(int, char**)
{
  constexpr int n = 2;
  constexpr int m = 1;
  using DDPController = controllers::DDPController<n, m>;

  using State_t = DDPController::State_t;
  using Control_t = DDPController::Control_t;

  auto calc_delta = [](const State_t& x,
                       const Control_t& u) -> State_t {
    State_t res;
    res << x(1), g / l * sin(x(0)) + u(0) - friction_coeff * x(1);
    return res;
  };

  auto dynamics = [=](const State_t& x, const Control_t& u) -> State_t {
    return kinematics::integrate(x, u, calc_delta, delta_t);
  };

  auto cost = [](const State_t& x, const Control_t& u) -> double {
    constexpr double alpha = 0.5;
    constexpr double beta = 0.9;
    double u_cost = alpha * alpha * (std::cosh(beta * u(0) / alpha) - 1);
    double x_cost = 1e3 * x(0) * x(0);
    return u_cost + x_cost;
  };

  auto final_cost = [](const State_t& x) -> double {
    return 1e4 * (x(0) * x(0)) + 1e2 * (x(1) * x(1));
  };

  DDPController controller(dynamics, cost, final_cost);


  State_t x;
  x << 2.5, 0.0;
  Eigen::Matrix<double, 1, 1> u{0.0};

  auto wrapped = utils::createFullFunction<n, m>(dynamics);
  auto full_state = utils::createFullState(x, u);
  auto result = numerics::quadratizeVectorFunction(wrapped, full_state);

  std::cout << "things:\n";
  for (const auto& thing : result)
  {
    std::cout << "\n" << thing << "\n";
  }

  auto controls = controller.solve(x, timesteps);
  std::cout << "\n\n>> DONE! <<\n\n";
  {
    std::ofstream csv("ddp.csv");
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
