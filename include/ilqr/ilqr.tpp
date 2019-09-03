#pragma once
#ifndef CONTROLS_TEST_ILQR_TPP
#define CONTROLS_TEST_ILQR_TPP

#include <Eigen/Dense>
#include "ilqr.h"

namespace controllers
{
template <int n, int m>
ILQRController<n, m>::ILQRController(const ILQRController::DynamicsFunction &dynamics_function,
                                     const ILQRController::CostFunction &cost_function) noexcept
  : dynamics_function_{ dynamics_function }, cost_function_{ cost_function }, lqr_controller_{}
{
}
template <int n, int m>
typename ILQRController<n, m>::ControlArray_t ILQRController<n, m>::solve(const ILQRController::State_t &x0,
                                                                          int timesteps) const
{
  constexpr double error = 1e-1;
  constexpr int iterations = 100;

  const int num_controls = timesteps - 1;

  ControlArray_t controls(m, num_controls);
  StateArray_t states(n, timesteps);
  states.col(0) = x0;

  forwardPropogate(states, controls);

  for (int i = 0; i < iterations; i++)
  {
    F_t F = approximateDynamics(states, controls);
  }

  return controls;
}

template <int n, int m>
void ILQRController<n, m>::forwardPropogate(ILQRController::StateArray_t &states,
                                            const ILQRController::ControlArray_t &controls) const
{
  int num_controls = controls.cols();
  for (int i = 0; i < num_controls; i++)
  {
    auto previous_state = states.col(i);
    auto control = controls.col(i);
    states.col(i + 1) = dynamics_function_(previous_state, control);
  }
}
template <int n, int m>
typename ILQRController<n, m>::FArray_t ILQRController<n, m>::approximateDynamics(
    const ILQRController::StateArray_t &states, const ILQRController::ControlArray_t &controls) const
{
  const int num_controls = controls.cols();

  FArray_t F_array(n, (n + m) * num_controls);

  auto wrapped = [=](const Full_t &full) {
    State_t x = full.head(n);
    Control_t u = full.tail(m);
    return dynamics_function_(x, u);
  };

  constexpr double epsilon = 1e-9;

  for (int t = 0; t < num_controls; t++)
  {
    auto F = F_array.template block<n, n + m>(0, (n + m) * t);
    Full_t delta = Full_t::Zero();

    Full_t x{};
    x.head(n) = states.col(t);
    x.tail(m) = controls.col(t);

    for (int i = 0; i < n + m; i++)
    {
      delta(i) = epsilon;
      State_t positive = wrapped(x + delta);
      State_t negative = wrapped(x - delta);
      F.col(i) = (positive - negative) / 2 * epsilon;
      delta(i) = 0.0;
    }
  }

  return F_array;
}
template <int n, int m>
typename ILQRController<n, m>::ApproximateCosts ILQRController<n, m>::approximateCost(
    const ILQRController::StateArray_t &states, const ILQRController::ControlArray_t &controls) const
{
  int num_controls = controls.cols();

  CArray_t C_array(n + m, (n + m) * num_controls);
  cArray_t c_array(n + m, num_controls);

  auto wrapped = [=](const Full_t &full) {
    State_t x = full.head(n);
    Control_t u = full.tail(m);
    return cost_function_(x, u);
  };

  constexpr double epsilon = 1e-9;

  for (int t = 0; t < num_controls; t++)
  {
    auto c = c_array.template block<n + m, 1>(0, t);
    Full_t delta = Full_t::Zero();

    Full_t x{};
    x.head(n) = states.col(t);
    x.tail(m) = controls.col(t);

    for (int i = 0; i < n + m; i++)
    {
      delta(i) = epsilon;
      State_t positive = wrapped(x + delta);
      State_t negative = wrapped(x - delta);
      c(i) = (positive - negative) / 2 * epsilon;
      delta(i) = 0.0;
    }
  }

  for (int t = 0; t < num_controls; t++)
  {
    auto C = C_array.template block<n + m, n + m>(0, (n + m) * t);
    Full_t d1 = Full_t::Zero();
    Full_t d2 = Full_t::Zero();

    Full_t x{};
    x.head(n) = states.col(t);
    x.tail(m) = controls.col(t);

    for (int i = 0; i < n + m; i++)
    {
      d1(i) = epsilon;

      for (int j = 0; j < n + m; j++)
      {
        d2(j) = epsilon;

        if (i == j)
        {
          C(i, j) = (-wrapped(x + 2 * d1) + 16 * wrapped(x + d1) - 30 * wrapped(x) + 16 * wrapped(x - d1) -
                     wrapped(x - 2 * d1)) /
                    (12 * epsilon * epsilon);
        }
        else
        {
          C(i, j) = (wrapped(x + d1 + d2) - wrapped(x + d1 - d2) - wrapped(x - d1 + d2) + wrapped(x - d1 - d2)) /
                    (4 * epsilon * epsilon);
        }

        d2(j) = 0.0;
      }
      d1(i) = 0.0;
    }
  }

  return { C_array, c_array };
}

}  // namespace controllers

#endif  // CONTROLS_TEST_ILQR_TPP
