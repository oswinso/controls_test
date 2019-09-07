#pragma once
#ifndef CONTROLS_TEST_DDP_TPP
#define CONTROLS_TEST_DDP_TPP

#include <numerics/approximation.h>
#include <utils/utils.h>
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

template <int n, int m>
std::vector<typename DDPController<n, m>::Control_t> DDPController<n, m>::solve(const DDPController::State_t &x0,
                                                                                int timesteps, int iterations)
{
  constexpr double convergence_threshold = 1e-15;

  const int num_controls = timesteps - 1;
  std::vector<Control_t> controls(num_controls, Control_t::Zero());
  std::vector<State_t> states(timesteps, x0);

  double cost = forwardPropogate(states, controls);

  printState(states, controls);
  for (int i = 0; i < iterations; i++)
  {
    // 1: Approximate Dynamics and Cost
    std::vector<Dynamics> dynamics = approximateDynamics(states, controls);
    auto [costs, final_cost] = approximateCosts(states, controls);

    // 2: Backwards Pass
    auto step_results = backwardsPass(timesteps, dynamics, costs, final_cost);

    // 3: Forward Pass
    bool forward_pass_done = false;
    double alpha = 1.0;
    constexpr int search_iterations = 10;
    for (int k = 0; k < search_iterations; k++)
    {
      std::vector<Control_t> new_controls(controls);
      std::vector<State_t> new_states(states);
      for (int j = 0; j < num_controls; j++)
      {
        auto delta_u = step_results[j].K * (new_states[j] - states[j]) + alpha * step_results[j].k;
        new_controls[j] += delta_u;
        new_states[j + 1] = dynamics_function_(new_states[j], new_controls[j]);
      }

      double new_cost = forwardPropogate(new_states, new_controls);
      if (std::isnan(new_cost))
      {
        std::cerr << "new_cost = NaN!\n";
        continue;
      }
      double d_cost = cost - new_cost;
      double expected_cost = -alpha * dV(0) - 0.5 * alpha * alpha * dV(1);
      assert(expected_cost != 0.0);
      double z = d_cost / expected_cost;

      bool iteration_converged = z > 0;
      bool solve_converged = i > 0 && abs(d_cost / cost) < convergence_threshold;

      if (iteration_converged)
      {
        std::cout << "Iteration " << i << " converged!\n";
        printf("%10s\t%10s\t%10s\t%10s\t%10s\n", "d_cost:", "old_cost:", "new_cost:", "expected_cost:", "z:");
        printf("%10e\t%10e\t%10e\t%10e\t%10e\n", d_cost, cost, new_cost, expected_cost, z);
        controls.swap(new_controls);
        states.swap(new_states);
        cost = new_cost;
        forward_pass_done = true;
        break;
      }

      if (solve_converged)
      {
        return new_controls;
      }
      std::cout << "Iteration " << i << " did not converge.\n";
      printf("%10s\t%10s\t%10s\t%10s\t%10s\n", "d_cost:", "old_cost:", "new_cost:", "expected_cost:", "z:");
      printf("%10e\t%10e\t%10e\t%10e\t%10e\n", d_cost, cost, new_cost, expected_cost, z);

      std::cout << "result:\n";
      for (const auto& u : new_controls)
      {
        std::cout << u(0) << ", ";
      }
      std::cout << "\n";
      for (const auto& x : new_states)
      {
        std::cout << "(" << x(0) << ", " << x(1) << ")  ";
      }
      std::cout << "\n\n";

      alpha = alpha / 2;
    }

    if (forward_pass_done)
    {
      decreaseMu();
    }
    else
    {
      increaseMu();
      i--; // Redo this iteration
    }
  }
  return controls;
}

template <int n, int m>
double DDPController<n, m>::forwardPropogate(std::vector<State_t> &states, const std::vector<Control_t> &controls) const
{
  double cost = 0.0;

  int num_controls = controls.size();
  for (int i = 0; i < num_controls; i++)
  {
    auto previous_state = states[i];
    auto control = controls[i];
    states[i + 1] = dynamics_function_(previous_state, control);

    cost += cost_function_(states[i], control);
  }
  cost += final_cost_function_(states.back());

  if (std::isnan(cost))
  {
    return std::numeric_limits<double>::infinity();
  }
  return cost;
}

template <int n, int m>
std::vector<typename DDPController<n, m>::Dynamics> DDPController<n, m>::approximateDynamics(
    const std::vector<State_t> &states, const std::vector<Control_t> &controls) const
{
  const int num_controls = controls.size();
  std::vector<Dynamics> dynamics_vector;
  dynamics_vector.reserve(num_controls);

  constexpr double epsilon = 1e-6;

  auto wrapped_dynamics_function = utils::createFullFunction<n, m>(dynamics_function_);

  for (int t = 0; t < num_controls; t++)
  {
    auto full_state = utils::createFullState(states[t], controls[t]);

    Dynamics dynamics{};
    dynamics.F = numerics::linearize<n>(wrapped_dynamics_function, full_state, epsilon);
    dynamics.F_hessian = numerics::quadratizeVectorFunction(wrapped_dynamics_function, full_state, epsilon);

    dynamics_vector.emplace_back(dynamics);
  }

  return dynamics_vector;
}

template <int n, int m>
std::pair<std::vector<typename DDPController<n, m>::Costs>, typename DDPController<n, m>::FinalCosts>
DDPController<n, m>::approximateCosts(const std::vector<State_t> &states, const std::vector<Control_t> &controls) const
{
  constexpr double epsilon = 1e-6;
  const int num_controls = controls.size();

  std::vector<Costs> costs_vector;
  costs_vector.reserve(num_controls);

  FinalCosts final_costs {
    numerics::quadratize(final_cost_function_, states[num_controls], epsilon),
    numerics::linearizeScalar(final_cost_function_, states[num_controls], epsilon).transpose()
  };

  auto wrapped_cost_function = utils::createFullFunction<n, m>(cost_function_);

  for (int t = 0; t < num_controls; t++)
  {
    auto full_state = utils::createFullState(states[t], controls[t]);

    Costs costs;
    costs.C = numerics::quadratize(wrapped_cost_function, full_state, epsilon);
    costs.c = numerics::linearizeScalar(wrapped_cost_function, full_state, epsilon);

    costs_vector.emplace_back(costs);
  }

  return std::make_pair(std::move(costs_vector), final_costs);
}

template <int n, int m>
void DDPController<n, m>::increaseMu()
{
  std::cerr << "Increasing (mu, delta) from (" << mu << ", " << delta << ") to ";
  delta = std::max(delta_zero, delta * delta);
  mu = std::max(mu_min, mu * delta);
  std::cerr << "(" << mu << ", " << delta << ")\n";

  if (std::isinf(mu))
  {
    std::cerr << "rip, mu is infinity. Aborting...\n";
    std::exit(1);
  }
}

template <int n, int m>
void DDPController<n, m>::decreaseMu()
{
  delta = std::min(1 / delta_zero, delta / delta_zero);
  mu = mu * delta > mu_min ? mu * delta : 0.0;
}
template <int n, int m>
std::vector<typename DDPController<n, m>::StepResult>
DDPController<n, m>::backwardsPass(int timesteps, const std::vector<Dynamics> &dynamics,
                                   const std::vector<Costs> &costs, const DDPController::FinalCosts &final_cost)
{
  const int num_controls = timesteps - 1;

  assert(static_cast<int>(dynamics.size()) == num_controls);
  assert(static_cast<int>(costs.size()) == num_controls);

  bool backwards_pass_done = false;

  std::vector<StepResult> step_results(num_controls);

  while (!backwards_pass_done)
  {
    dV.setZero();

    V_t V = final_cost.V_final;
    v_t v = final_cost.v_final;

    for (int t = num_controls - 1; t >= 0; t--)
    {
      if (auto result = backwardsStep(V, v, costs[t], dynamics[t]); result)
      {
        step_results[t] = *result;
      }
      else
      {
        increaseMu();
        break;
      }

      if (t == 0)
      {
        backwards_pass_done = true;
      }
    }
  }

  return step_results;
}
template <int n, int m>
std::optional<typename DDPController<n, m>::StepResult>
DDPController<n, m>::backwardsStep(DDPController::V_t &V, DDPController::v_t &v, const DDPController::Costs &costs,
                                   const DDPController::Dynamics &dynamics)
{
  Q_t Q = costs.C + dynamics.F.transpose() * V * dynamics.F;
  q_t q = costs.c + dynamics.F.transpose() * v;

  Eigen::Ref<Quu_t> Quu = Q.template block<m, m>(n, n);
  Quu_t Quu_inverse = getRegularizedInverse(Quu);

  K_t K = -Quu_inverse * getxu<n, m>(Q).transpose();
  k_t k = -Quu_inverse * getu<n, m>(q);

  dV(0) += 0.5 * getu<n, m>(q) * k;
  dV(1) += 0.0;

  V = getxx<n, m>(Q) + K.transpose() * getuu<n, m>(Q) * K + 2 * getxu<n, m>(Q) * K;
  v = getx<n, m>(q) + K.transpose() * getuu<n, m>(Q) * k + K.transpose() * getu<n, m>(q) + getxu<n, m>(Q) * k;

  StepResult step_result{ K, k };
  return step_result;
}
template <int n, int m>
typename DDPController<n, m>::Quu_t DDPController<n, m>::getRegularizedInverse(const Eigen::Ref<Quu_t> &Quu) const
{
  Eigen::SelfAdjointEigenSolver<Quu_t> eigen_solver(Quu);
  assert(eigen_solver.info() == Eigen::Success);
  Eigen::Matrix<double, m, 1> eigenvalues = eigen_solver.eigenvalues();
  for (int i = 0; i < eigenvalues.size(); i++)
  {
    eigenvalues[i] = std::max(eigenvalues[i], 0.0);
    eigenvalues[i] = 1.0 / (eigenvalues[i] + mu);
  }
  return eigen_solver.eigenvectors() * eigenvalues.asDiagonal() * eigen_solver.eigenvectors().transpose();
}
template<int n, int m>
void DDPController<n, m>::printState(const std::vector<State_t> &states, const std::vector<Control_t> &controls) const
{
  std::cout << ">>>> STATES:\n";
  for (const auto &x : states)
  {
    std::cout << "(" << x.transpose() << ") ";
  }
  std::cout << "\n\n>>>> CONTROLS:\n";
  for (const auto &u : controls)
  {
    std::cout << "(" << u.transpose() << ") ";
  }
  std::cout << "\n\n";
}
}  // namespace controllers

#endif  // CONTROLS_TEST_DDP_TPP
