#pragma once
#ifndef CONTROLS_TEST_LQR_TPP
#define CONTROLS_TEST_LQR_TPP

#include <Eigen/Dense>
#include "lqr.h"

namespace controllers
{
template <int n, int m>
LQRController<n, m>::LQRController(const controllers::LQRController<n, m>::Dynamics &dynamics,
                                   const controllers::LQRController<n, m>::Costs &costs) noexcept
  : dynamics_{ dynamics }, costs_{ costs }
{
}

template <int n, int m>
void LQRController<n, m>::setDynamics(const Dynamics& dynamics) noexcept
{
  dynamics_ = dynamics;
}

template <int n, int m>
void LQRController<n, m>::setCosts(const Costs& costs) noexcept
{
  costs_ = costs;
}

template <int n, int m>
typename LQRController<n, m>::ControlArray_t LQRController<n, m>::solve(const LQRController::State_t &x0,
                                                                        int timesteps) const
{
  int num_controls = timesteps - 1;
  Eigen::Matrix<double, m, Eigen::Dynamic> Ks(m, n * num_controls);
  Eigen::Matrix<double, m, Eigen::Dynamic> ks(m, num_controls);

  V_t V = getxx(costs_.C);
  v_t v = getx(costs_.c);
  for (int i = num_controls - 1; i >= 0; i--)
  {
    Q_t Q = costs_.C + dynamics_.F.transpose() * V * dynamics_.F;
    q_t q = costs_.c + dynamics_.F.transpose() * (V * dynamics_.f + v);

    Eigen::Matrix<double, m, m> Quu_inverse = Eigen::Matrix<double, m, m>(getuu(Q)).inverse();

    auto K = Ks.template block<m, n>(0, i * n);
    auto k = ks.template block<m, 1>(0, i);

    K = -Quu_inverse * getxu(Q).transpose();
    k = -Quu_inverse * getu(q);

    V = getxx(Q) + 2 * getxu(Q) * K + K.transpose() * getuu(Q) * K;
    v = getx(q) + getxu(Q) * k + K.transpose() * getu(q) + K.transpose() * getuu(Q) * k;
  }

  ControlArray_t controls(m, num_controls);
  State_t x = x0;
  for (int i = 0; i < num_controls; i++)
  {
    auto K = Ks.template block<m, n>(0, i * n);
    auto k = ks.template block<m, 1>(0, i);

    auto u = controls.col(i);

    u = K * x + k;
    x = dynamics_.propogate(x, u);
  }

  return controls;
}

template <int n, int m>
auto LQRController<n, m>::getxx(const Eigen::Matrix<double, n + m, n + m> &matrix) const
{
  return matrix.template block<n, n>(0, 0);
}

template <int n, int m>
auto LQRController<n, m>::getxu(const Eigen::Matrix<double, n + m, n + m> &matrix) const
{
  return matrix.template block<n, m>(0, n);
}

template <int n, int m>
auto LQRController<n, m>::getuu(const Eigen::Matrix<double, n + m, n + m> &matrix) const
{
  return matrix.template block<m, m>(n, n);
}

template <int n, int m>
auto LQRController<n, m>::getx(const Eigen::Matrix<double, n + m, 1> &matrix) const
{
  return matrix.template block<n, 1>(0, 0);
}

template <int n, int m>
auto LQRController<n, m>::getu(const Eigen::Matrix<double, n + m, 1> &matrix) const
{
  return matrix.template block<m, 1>(n, 0);
}

template <int n, int m>
typename LQRController<n, m>::State_t LQRController<n, m>::Dynamics::propogate(const LQRController::State_t &x,
                                                                               const LQRController::Control_t &u) const
{
  Eigen::Matrix<double, n + m, 1> big_vector;
  big_vector.template head<n>() = x;
  big_vector.template tail<m>() = u;
//
//  std::cout << "torque = g/l sin(theta) ~ " << F(1, 0) << " * " << x(0) << " + " << f(1) << " = "
//            << F(1, 0) * x(0) + f(1) << std::endl;

  return F * big_vector + f;
}
}  // namespace controllers

#endif  // CONTROLS_TEST_LQR_TPP
