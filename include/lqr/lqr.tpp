#pragma once
#ifndef CONTROLS_TEST_LQR_TPP
#define CONTROLS_TEST_LQR_TPP

#include <iostream>
#include <Eigen/Dense>
#include "lqr.h"

namespace controllers
{
template <int n, int m>
LQRController<n, m>::LQRController() noexcept
{
}

template <int n, int m>
LQRController<n, m>::LQRController(const Dynamics &dynamics, const Costs &costs, const FinalCosts &final_costs) noexcept
  : dynamics_{ dynamics }, costs_{ costs }, final_costs_{ final_costs }
{
}

template <int n, int m>
void LQRController<n, m>::setDynamics(const Dynamics &dynamics) noexcept
{
  dynamics_ = dynamics;
}

template <int n, int m>
void LQRController<n, m>::setCosts(const Costs &costs) noexcept
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

  V_t V = final_costs_.V_final;
  v_t v = final_costs_.v_final;
  for (int i = num_controls - 1; i >= 0; i--)
  {
    auto [K, k] = LQRController<n, m>::lqrStep(V, v, costs_, dynamics_);
    Ks.template block<m, n>(0, i * n) = K;
    ks.template block<m, 1>(0, i) = k;
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
std::vector<typename LQRController<n, m>::LQRStepResult>
LQRController<n, m>::solve(int timesteps, const std::vector<Dynamics> &dynamics,
                           const std::vector<Costs> &costs, const FinalCosts &final_costs) const
{
  int num_controls = timesteps - 1;

  assert(static_cast<int>(dynamics.size()) == num_controls);
  assert(static_cast<int>(costs.size()) == num_controls);

  V_t V = final_costs.V_final;
  v_t v = final_costs.v_final;
//  std::cout << "V:\n" << V << "\nv:\n" << v << "\n";
  std::vector<LQRStepResult> step_results(num_controls);
  for (int i = num_controls - 1; i >= 0; i--)
  {
    step_results[i] = lqrStep(V, v, costs[i], dynamics[i]);
    std::cout << "K: " << step_results[i].K << ", k: " << step_results[i].k << "\n";
  }

  return step_results;
}

template <int n, int m>
auto LQRController<n, m>::getxx(const Eigen::Matrix<double, n + m, n + m> &matrix)
{
  return matrix.template block<n, n>(0, 0);
}

template <int n, int m>
auto LQRController<n, m>::getxu(const Eigen::Matrix<double, n + m, n + m> &matrix)
{
  return matrix.template block<n, m>(0, n);
}

template <int n, int m>
auto LQRController<n, m>::getuu(const Eigen::Matrix<double, n + m, n + m> &matrix)
{
  return matrix.template block<m, m>(n, n);
}

template <int n, int m>
auto LQRController<n, m>::getx(const Eigen::Matrix<double, n + m, 1> &matrix)
{
  return matrix.template block<n, 1>(0, 0);
}

template <int n, int m>
auto LQRController<n, m>::getu(const Eigen::Matrix<double, n + m, 1> &matrix)
{
  return matrix.template block<m, 1>(n, 0);
}

template <int n, int m>
typename LQRController<n, m>::LQRStepResult LQRController<n, m>::lqrStep(V_t &V, v_t &v,
                                                                         const LQRController::Costs &costs,
                                                                         const LQRController::Dynamics &dynamics)
{
  Q_t Q = costs.C + dynamics.F.transpose() * V * dynamics.F;
  q_t q = costs.c + dynamics.F.transpose() * (V * dynamics.f + v);

  std::cout << "Quu: " << getuu(Q) << ", Qxu: " << getxu(Q).transpose() << ", qu: " << getu(q).transpose() << ", F: " << dynamics.F.transpose() << ", f: " << dynamics.f << "\n";
  Eigen::Matrix<double, m, m> Quu_inverse = Eigen::Matrix<double, m, m>(getuu(Q)).inverse();

  K_t K = -Quu_inverse * getxu(Q).transpose();
  k_t k = -Quu_inverse * getu(q);

  V = getxx(Q) + 2 * getxu(Q) * K + K.transpose() * getuu(Q) * K;
  v = getx(q) + getxu(Q) * k + K.transpose() * getu(q) + K.transpose() * getuu(Q) * k;

  return { K, k };
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
