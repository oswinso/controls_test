#include <Eigen/Core>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <lqr/lqr.h>
#include <kinematics/rk4.h>

constexpr double delta_t = 1e-2;
constexpr double g = 9.80665;
constexpr double l = 1.0;
constexpr int timesteps = 1000;
constexpr double friction_coeff = 0.1;
constexpr double max_torque = 100.0;

void test()
{
  using LQRController = controllers::LQRController<2, 1>;
  // clang-format off
  LQRController::State_t x{2.5, 0.0};

  double angle = x(0);
  LQRController::F_t F;
  F <<  1.0,                         delta_t, 0.0,
        g/l * cos(angle) * delta_t,  1.0,     delta_t;

  LQRController::f_t f;
  f << 0.0, g/l * (sin(angle) - angle * cos(angle)) * delta_t;

  LQRController::C_t C;
  C <<  1e6, 0.0, 0.0,
        0.0, 1e1, 0.0,
        0.0, 0.0, 1e4;

  LQRController::v_t v_final;
  v_final << 0.0, 0.0;

  LQRController::V_t V_final;
  V_final <<  1e10, 0.0,
              0.0,  1e1;

  LQRController::c_t c;
  c << 0.0, 0.0, 0.0;
  // clang-format on

  LQRController::Dynamics dynamics{ F, f };
  LQRController::Costs costs{ C, c, V_final, v_final};
  LQRController controller(dynamics, costs);

  {
    using kinematics::integrate;
    std::ofstream csv("lqr.csv");
    csv << "u,x0,x1\n";

    csv << "0," << x(0) << "," << x(1) << "\n";

    auto calcDelta = [](const LQRController::State_t& x, const LQRController::Control_t& u) -> LQRController::State_t {
      LQRController::State_t res;
      res << x(1), g / l * sin(x(0)) + u(0) - friction_coeff * x(1);
      return res;
    };

    for (int i = 0; i < timesteps; i++)
    {
      double angle = x(0);
      F <<  1.0,                         delta_t, 0.0,
        g/l * cos(angle) * delta_t,  1.0,     delta_t;

      f << 0.0, g/l * (sin(angle) - angle * cos(angle)) * delta_t;
      dynamics = {F, f};
      controller.setDynamics(dynamics);

      auto res = controller.solve(x, timesteps);
      LQRController::Control_t u  = res.col(0);
//      u(0) = std::clamp(u(0), -max_torque, max_torque);
      x = integrate(x, u, calcDelta, delta_t);
//      Eigen::Vector3d full{x(0), x(1), u(0)};
//      x = F * full + f;
      std::cout << u(0) << "," << x(0) << "," << x(1) << "\n";
      csv << u(0) << "," << x(0) << "," << x(1) << "\n";
    }
  }
}

int main()
{
  test();
  return 0;
}
