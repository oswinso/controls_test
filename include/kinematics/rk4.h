#ifndef CONTROLS_TEST_RK4_H
#define CONTROLS_TEST_RK4_H

#include <iostream>
#include <Eigen/Core>

namespace kinematics
{
template <int n, int m, typename DeltaX>
Eigen::Matrix<double, n, 1> integrate(const Eigen::Matrix<double, n, 1>& x, const Eigen::Matrix<double, m, 1>& u,
                                      const DeltaX& f, double delta_t)
{
//  std::cout << "f(x, u): " << f(x, u)(1) << ". gravity: " << f(x, Eigen::Matrix<double, m, 1>::Zero())(1)
//            << ". u (" << u << "): " << f(Eigen::Matrix<double, n, 1>::Zero(), u)(1) << std::endl;
  Eigen::Matrix<double, n, 1> k1 = f(x, u);
  Eigen::Vector2d k2 = f(x + k1 * delta_t / 2.0, u);
  Eigen::Vector2d k3 = f(x + k2 * delta_t / 2.0, u);
  Eigen::Vector2d k4 = f(x + k3 * delta_t, u);

  return x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0 * delta_t;
}
}  // namespace kinematics

#endif  // CONTROLS_TEST_RK4_H
