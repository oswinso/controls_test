#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <fstream>

constexpr double delta_t = 1e-1;
constexpr double g = 9.80665;
constexpr double l = 1.0;
constexpr int timesteps = 10;

using VectorVector2d = std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>;

Eigen::Vector2d propogate(const Eigen::Vector2d& state, double control)
{
  std::cout << "state: " << state.transpose() << std::endl;
  double theta = state(0);
  double theta_dot = state(1);

  double theta_ddot = -g / l * sin(theta) + control;
  std::cout << "theta_ddot: " << theta_ddot << std::endl;
  theta += theta_dot * delta_t + 0.5 * theta_ddot * delta_t * delta_t;
  theta_dot += theta_ddot * delta_t;

  return {theta, theta_dot};
}

std::vector<double> getControl(const Eigen::Vector2d& x)
{
//  std::cout << ">> getControl" << std::endl;
  // Define Variables
  double phi = x(0);
  Eigen::Vector3d state{};
  state.head<2>() = x;
  state(2) = 1.0;

  Eigen::Matrix3d a;
  // clang-format off
  a <<  1,                          delta_t,  0,
        -g/l * cos(phi) * delta_t,  1,        -g/l * (sin(phi) - phi * cos(phi)),
        0,                          0,        1;
  // clang-format on

  Eigen::Vector3d b{ 0, delta_t, 0 };

  Eigen::Matrix3d q;
  // clang-format off
  q <<  1e5,  0,    0,
        0,    1e5,  0,
        0,    0,    0;
  // clang-format on

  constexpr double r = 0.0;

  Eigen::MatrixX3d ks(timesteps - 1, 3);

  Eigen::Matrix3d p = q;
  // Start backwards dynamic programming
  for (int i = timesteps - 2; i >= 0; i--)
  {
    Eigen::Matrix3d q_xx = a.transpose() * p * a + q;
    Eigen::Matrix<double, 1, 3> q_xu = b.transpose() * p * a;
    double q_uu = r + b.transpose() * p * b;

    Eigen::Matrix<double, 1, 3> k = -1 / q_uu * q_xu;
    ks.row(i) = k;
//    std::cout << "row " << i << ": " << k << "\n";

    p = q_xx + q_xu.transpose() * k;
  }

  std::vector<double> controls;
  controls.reserve(timesteps - 1);

//  std::cout << ">> forward step" << std::endl;
  for (int i = 0; i < timesteps - 1; i++)
  {
//    std::cout << "row " << i << ": " << ks.row(i) << "\n";
//    std::cout << "row " << i << " * state: " << ks.row(i) * state << "\n";
    double u = (ks.row(i) * state)(0);
    controls.emplace_back(u);
    state = a * state + b * u;
  }

  return controls;
}

int main()
{
  // x = [θ  dθ/dt  1]^T
  //
  //     | 1              Δt  0                     |
  // A = | -g/l cosφ Δt   1   -g/l (sinφ - φ cosφ)  |
  //     | 0              0   1                     |
  //
  // B = [0 Δt 0]^T

  // Qxx = A^T P_k+1 A + Q
  // Qxu = B^T P A
  // Quu = R + B^T P B

  // K = -Quu^-1 Qux
  // u_k = K x_k

  // P_k = Qxx + Qxu K

  std::cout << "\n";
  std::cout << "========================" << std::endl;
  std::cout << "      No controls       " << std::endl;
  std::cout << "========================" << std::endl;
  {
    std::ofstream csv("no_controls.csv");
    csv << "u,x0,x1\n";

    Eigen::Vector2d x0{ 1.0, 0.0 };
    csv << "0," << x0(0) << "," << x0(1) << "\n";

    std::cout << "Starting: (" << x0(0) << ", " << x0(1) << ")" << std::endl;
    for (int i = 0; i < timesteps; i++)
    {
      x0 = propogate(x0, 0.0);
      std::cout << "Control: " << 0.0 << ", state: (" << x0(0) << ", " << x0(1) << ")" << std::endl;
      csv << 0.0 << "," << x0(0) <<"," << x0(1) << "\n";
    }
  }
//
//  std::cout << "\n";
//  std::cout << "========================" << std::endl;
//  std::cout << "      Executing all     " << std::endl;
//  std::cout << "========================" << std::endl;
//  {
//    Eigen::Vector2d x0{ 1.0, 0.0 };
//    std::vector<double> us = getControl(x0);
//
//    std::cout << "Starting: (" << x0(0) << ", " << x0(1) << ")" << std::endl;
//    for (const auto u : us)
//    {
//      x0 = propogate(x0, u);
//      std::cout << "Control: " << u << ", state: (" << x0(0) << ", " << x0(1) << ")" << std::endl;
//    }
//
//  }
//  std::cout << "\n";
//  std::cout << "========================" << std::endl;
//  std::cout << "Now executing first only" << std::endl;
//  std::cout << "========================" << std::endl;
//
//  {
//    Eigen::Vector2d x0 {1.0, 0.0};
//    std::cout << "Starting: (" << x0(0) << ", " << x0(1) << ")" << std::endl;
//
//    for (int i = 0; i < timesteps - 1; i++)
//    {
//      std::vector<double> us = getControl(x0);
//      double u = us[0];
//      x0 = propogate(x0, u);
//      std::cout << "Control: " << u << ", state: (" << x0(0) << ", " << x0(1) << ")" << std::endl;
//    }
//  }

//  for (size_t i = 0; i < controls.size(); i++)
//  {
//    std::cout << "Control: " << controls[i] << ", state: (" << states[i](0) << ", " << states[i](1) << ")" << std::endl;
//  }

  return 0;
}
