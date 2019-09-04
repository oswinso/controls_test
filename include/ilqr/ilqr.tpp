#pragma once
#ifndef CONTROLS_TEST_ILQR_TPP
#define CONTROLS_TEST_ILQR_TPP

#include <iostream>
#include <numerics/approximation.h>
#include <Eigen/Dense>
#include "ilqr.h"

namespace controllers
{
template <int n, int m>
ILQRController<n, m>::ILQRController(const DynamicsFunction &dynamics_function, const CostFunction &cost_function,
                                     const FinalCostFunction &final_cost_function) noexcept
  : dynamics_function_{ dynamics_function }
  , cost_function_{ cost_function }
  , final_cost_function_{ final_cost_function }
{
}

template <int n, int m>
std::vector<typename ILQRController<n, m>::Control_t> ILQRController<n, m>::solve(const ILQRController::State_t &x0,
                                                                                  int timesteps) const
{
  //  constexpr double error = 1e-1;
  constexpr int iterations = 20;

  const int num_controls = timesteps - 1;

  Control_t one;
  one << -10.0;
  std::vector<Control_t> controls(num_controls, one);
  std::vector<State_t> states(timesteps, x0);

  forwardPropogate(states, controls);

  //  double cost = std::numeric_limits<double>::infinity();

  std::cout << "begin:\n";
  printState(states, controls);

  for (int i = 0; i < iterations; i++)
  {
    std::vector<typename LQR::Dynamics> dynamics = approximateDynamics(states, controls);
    auto [costs, final_cost] = approximateCost(states, controls);

    State_t x_t = states.front();
    auto step_results = lqr_controller_.solve(timesteps, dynamics, costs, final_cost);

    std::cout << "i:"<<i<<"\n";
    for (int j = 0; j < num_controls; j++)
    {
      auto delta_u = step_results[j].K * (x_t - states[j]) + step_results[j].k;
      controls[j] += delta_u;
      std::cout << "K:" << step_results[j].K << "\n, x_t - states[j]: " << (x_t - states[j]).transpose() << ", k: " << step_results[j].k << ", delta_u: " << delta_u << "\n";
      x_t = dynamics_function_(x_t, controls[j]);
    }
    std::cout << "\n";
    forwardPropogate(states, controls);
    printState(states, controls);
  }

  return controls;
}

template <int n, int m>
void ILQRController<n, m>::forwardPropogate(std::vector<State_t> &states, const std::vector<Control_t> &controls) const
{
  int num_controls = controls.size();
  for (int i = 0; i < num_controls; i++)
  {
    auto previous_state = states[i];
    auto control = controls[i];
    states[i + 1] = dynamics_function_(previous_state, control);
  }
}

template <int n, int m>
std::vector<typename LQRController<n, m>::Dynamics> ILQRController<n, m>::approximateDynamics(
  const std::vector<State_t> &states, const std::vector<Control_t> &controls) const
{
  const int num_controls = controls.size();

  std::vector<typename LQR::Dynamics> F_array;
  F_array.reserve(num_controls);

  auto wrapped = [=](const Full_t &full) {
    State_t x = full.head(n);
    Control_t u = full.tail(m);
    return dynamics_function_(x, u);
  };

  constexpr double epsilon = 1e-9;

  for (int t = 0; t < num_controls; t++)
  {
    Full_t x{};
    x.head(n) = states[t];
    x.tail(m) = controls[t];

    typename LQR::Dynamics dynamics;
    dynamics.F = numerics::linearize<n>(wrapped, x, epsilon);
    dynamics.f = LQR::f_t::Zero();
    F_array.emplace_back(dynamics);
  }

  return F_array;
}

template <int n, int m>
std::pair<std::vector<typename LQRController<n, m>::Costs>, typename LQRController<n, m>::FinalCosts>
ILQRController<n, m>::approximateCost(const std::vector<State_t> &states, const std::vector<Control_t> &controls) const
{
  constexpr double epsilon = 1e-9;

  int num_controls = controls.size();

  std::vector<typename LQR::Costs> costs_vector(num_controls);

  typename LQR::FinalCosts final_costs{
    numerics::quadratize(final_cost_function_, states[num_controls], epsilon),
    numerics::linearizeScalar(final_cost_function_, states[num_controls], epsilon).transpose()
  };

  auto wrapped = [=](const Full_t &full) -> double {
    State_t x = full.head(n);
    Control_t u = full.tail(m);
    return cost_function_(x, u);
  };

  for (int t = 0; t < num_controls; t++)
  {
    Full_t x{};
    x.head(n) = states[t];
    x.tail(m) = controls[t];

    costs_vector[t].c = numerics::linearizeScalar(wrapped, x, epsilon).transpose();
  }

  for (int t = 0; t < num_controls; t++)
  {
    Full_t x{};
    x.head(n) = states[t];
    x.tail(m) = controls[t];

    costs_vector[t].C = numerics::quadratize(wrapped, x, epsilon);
  }

  return std::make_pair(std::move(costs_vector), final_costs);
}

template<int n, int m>
void ILQRController<n, m>::printState(const std::vector<State_t> &states, const std::vector<Control_t> &controls) const
{
  std::cout << ">>>> STATES:\n";
  for (const auto& x : states)
  {
    std::cout << "(" << x.transpose() << ") ";
  }
  std::cout << "\n\n>>>> CONTROLS:\n";
  for (const auto& u : controls)
  {
    std::cout << "(" << u.transpose() << ") ";
  }
  std::cout << "\n\n";
}
template<int n, int m>
void ILQRController<n, m>::printState(const std::vector<State_t>& states, const std::vector<Control_t>& controls, const std::vector<Control_t> control_deltas) const
{
  printState(states, controls);
  std::cout << ">>>> CONTROL DELTAS:\n";
  for (const auto& u : control_deltas)
  {
    std::cout << "(" << u.transpose() << ") ";
  }
  std::cout << "\n\n";
}

}  // namespace controllers

#endif  // CONTROLS_TEST_ILQR_TPP
