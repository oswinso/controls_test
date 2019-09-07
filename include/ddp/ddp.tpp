#pragma once
#ifndef CONTROLS_TEST_DDP_TPP
#define CONTROLS_TEST_DDP_TPP

#include <numerics/approximation.h>
#include <Eigen/Dense>
#include <iostream>
#include "ddp.h"

namespace controllers
{
template <int n, int m>
DDPController<n, m>::DDPController(const DynamicsFunction &dynamics_function, const CostFunction &cost_function,
                                     const FinalCostFunction &final_cost_function) noexcept
  : dynamics_function_{ dynamics_function }
  , cost_function_{ cost_function }
  , final_cost_function_{ final_cost_function }
{
}
template<int n, int m>
std::vector<typename DDPController<n, m>::Control_t> DDPController<n, m>::solve(const DDPController::State_t &x0, int timesteps)
{
  return std::vector<Control_t>();
}
}  // namespace controllers

#endif  // CONTROLS_TEST_DDP_TPP
