#pragma once
#ifndef CONTROLS_TEST_ILQR_TPP
#define CONTROLS_TEST_ILQR_TPP

#include <numerics/approximation.h>
#include <Eigen/Dense>
#include <iostream>
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
                                                                                  int timesteps)
{
  //  constexpr double error = 1e-1;
  constexpr int iterations = 100;

  const int num_controls = timesteps - 1;

  Control_t one;
  one << 0.0;
  std::vector<Control_t> controls(num_controls, one);
  std::vector<State_t> states(timesteps, x0);

  double cost = forwardPropogate(states, controls);

  //  double cost = std::numeric_limits<double>::infinity();

  //  std::cout << "begin:\n";
  //  printState(states, controls);

  for (int i = 0; i < iterations; i++)
  {
    // 1: Approximate Dynamics and Cost
    std::vector<typename LQR::Dynamics> dynamics = approximateDynamics(states, controls);
    auto [costs, final_cost] = approximateCost(states, controls);

    //    std::cout << "\n\n>>Finished Approximations<<\n";
    //    std::cout << "Last F:\n" << dynamics.back().F << "\n\n";
    //
    //    auto second_last = states[states.size() - 2];
    //    auto last = states[states.size() - 1];
    //    std::cout << "Second last x: " << second_last.transpose() << "\n";
    //    std::cout << "Last x: " << last.transpose() << "\n";
    //    std::cout << "Δx: " << (last - second_last).transpose() << "\n\n";
    //
    //    Control_t changed_controls = controls.back() + Control_t{ -1 };
    //    auto changed_last = dynamics_function_(second_last, changed_controls);
    //    std::cout << "Changed last: " << changed_last.transpose() << "\n";
    //    std::cout << "Changed Δx: " << (changed_last - second_last).transpose() << "\n";
    //    std::cout << "changed_last - last: " << (changed_last - last).transpose() << "\n\n";
    //
    //    Full_t second_last_full;
    //    second_last_full << 0.0, 0.0, -1.0;
    //    std::cout << "Fx second last:" << (dynamics.back().F * second_last_full).transpose() << "\n\n";
    //
    //    Full_t actual_second_last_full;
    //    actual_second_last_full.template head<n>() = second_last;
    //    actual_second_last_full.template tail<m>() = controls.back();
    //
    //    // Controls +delta and -delta
    //    double epsilon = 1e-9;
    //    {
    //      std::cout << "\n>>> epsilon=1e-9 <<<\n";
    //      auto original_controls = controls.back();
    //      auto original = dynamics_function_(second_last, original_controls);
    //      auto positive = dynamics_function_(second_last, original_controls + Control_t{ epsilon });
    //      auto negative = dynamics_function_(second_last, original_controls - Control_t{ epsilon });
    //
    //      std::cout << "Original:\t\t" << (original).transpose() << "\n";
    //      std::cout << "Positive:\t\t" << (positive).transpose() << "\n";
    //      std::cout << "Negative:\t\t" << (negative).transpose() << "\n";
    //      std::cout << "Differen:\t\t" << (positive - negative).transpose() << "\n";
    //      std::cout << "Derivati:\t\t" << ((positive - negative) / (2 * epsilon)).transpose() << "\n";
    //    }
    //    std::cout << "\n";
    //    epsilon = 1e-3;
    //    {
    //      std::cout << "\n>>> epsilon=1e-3 <<<\n";
    //      auto original_controls = controls.back();
    //      auto original = dynamics_function_(second_last, original_controls);
    //      auto positive = dynamics_function_(second_last, original_controls + Control_t{ epsilon });
    //      auto negative = dynamics_function_(second_last, original_controls - Control_t{ epsilon });
    //
    //      std::cout << "Original:\t\t" << (original).transpose() << "\n";
    //      std::cout << "Positive:\t\t" << (positive).transpose() << "\n";
    //      std::cout << "Negative:\t\t" << (negative).transpose() << "\n";
    //      std::cout << "Differen:\t\t" << (positive - negative).transpose() << "\n";
    //      std::cout << "Derivati:\t\t" << ((positive - negative) / (2 * epsilon)).transpose() << "\n";
    //    }
    //
    //    std::cout << "\n=======================================================================\n\n";

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
        //        std::cout << "K:" << step_results[j].K
        //                  << ", new_states[j] - states[j]: " << (new_states[j] - states[j]).transpose()
        //                  << ", k: " << step_results[j].k << ", delta_u: " << delta_u << "\n";
        new_states[j + 1] = dynamics_function_(new_states[j], new_controls[j]);
      }
      //      std::cout << "\n";

      double new_cost = forwardPropogate(new_states, new_controls);
      if (std::isnan(new_cost))
      {
        std::cerr << "new_cost is NaN!\n";
        continue;
      }
      double d_cost = cost - new_cost;
      double expected_cost = -alpha * dV(0) - 0.5 * alpha * alpha * dV(1);
      assert(expected_cost != 0.0);
      double z = d_cost / expected_cost;
      //      printf("%10s\t%10s\t%10s\t%10s\t%10s\n", "d_cost:", "old_cost:", "new_cost:", "expected_cost:", "z:");
      //      printf("%10e\t%10e\t%10e\t%10e\t%10e\n", d_cost, cost, new_cost, expected_cost, z);

      // Accept iteration if 0.0 < c1 < z
      if (z > 0)
      {
        std::cout << "Iteration " << i << " converged!\n";
        printf("%10s\t%10s\t%10s\t%10s\t%10s\n", "d_cost:", "old_cost:", "new_cost:", "expected_cost:", "z:");
        printf("%10e\t%10e\t%10e\t%10e\t%10e\n", d_cost, cost, new_cost, expected_cost, z);
        printState(states, controls);

        controls = new_controls;
        states = new_states;
        cost = new_cost;
        forward_pass_done = true;

        break;
      }

      alpha = alpha / 2;
    }

    if (forward_pass_done)
    {
      // Decrease mu since successful iteration
      decreaseMu();
    }
    else
    {
      // Forward pass failed, increase mu
      increaseMu();
    }
    //    std::cout << "i:" << i << "\n";
    //    printState(states, controls);
  }

  return controls;
}

template <int n, int m>
std::vector<typename ILQRController<n, m>::LQRStepResult>
ILQRController<n, m>::backwardsPass(int timesteps, const std::vector<Dynamics> &dynamics,
                                    const std::vector<Costs> &costs, const FinalCosts &final_costs)
{
  int num_controls = timesteps - 1;

  assert(static_cast<int>(dynamics.size()) == num_controls);
  assert(static_cast<int>(costs.size()) == num_controls);

  bool backwards_pass_done = false;

  std::vector<LQRStepResult> step_results(num_controls);

  while (!backwards_pass_done)
  {
    // reset dV
    dV.setZero();

    V_t V = final_costs.V_final;
    v_t v = final_costs.v_final;

    //  std::cout << "V:\n" << V << "\nv:\n" << v << "\n";
    for (int i = num_controls - 1; i >= 0; i--)
    {
      if (auto result = backwardsStep(V, v, costs[i], dynamics[i]); result)
      {
        step_results[i] = *result;
      }
      else
      {
        // Increase mu since Quu not PD
        std::cout << "Increasing (mu, delta) from (" << mu << ", " << delta << ") to ";
        increaseMu();
        std::cout << "(" << mu << ", " << delta << ")\n";
        break;
      }
      if (i == 0)
      {
        backwards_pass_done = true;
      }
    }
  }

  return step_results;
}

template <int n, int m>
typename ILQRController<n, m>::StepResult ILQRController<n, m>::backwardsStep(V_t &V, v_t &v, const Costs &costs,
                                                                              const Dynamics &dynamics)
{
  Q_t Q = costs.C + dynamics.F.transpose() * V * dynamics.F;
  q_t q = costs.c + dynamics.F.transpose() * v;
  //  std::cout << "F^T*v: " << (dynamics.F.transpose() * v).transpose() << ", c: " << costs.c.transpose()
  //            << ", v: " << v.transpose() << ", q: " << q.transpose() << "\n";

  Eigen::Ref<Eigen::Matrix<double, m, m>> Quu = Q.template block<m, m>(n, n);
  //  Eigen::Ref<Eigen::Matrix<double, n, m>> Qxu = Q.template block<n, m>(0, n);
  //  auto Cuu = getuu<n, m>(costs.C);
  //  auto Cxu = getxu<n, m>(costs.C);
  //  auto Fu = dynamics.F.template block<n, m>(0, n);
  //  auto Fx = dynamics.F.template block<n, n>(0, 0);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, m, m>> eigenSolver(Quu);
  assert(eigenSolver.info() == Eigen::Success);
  Eigen::Matrix<double, m, 1> eigenValues = eigenSolver.eigenvalues();
  for (int i = 0; i < eigenValues.size(); i++)
  {
    if (eigenValues[i] < 0.0)
    {
      eigenValues[i] = 0.0;
    }
    eigenValues[i] = 1.0 / (eigenValues[i] + mu);
  }
//  std::cout << "\n == \n";
//  std::cout << "V:\n" << V << "\n";
//  std::cout << "v: " << v.transpose() << "\t";
//  std::cout << "Quu: " << Quu;
  Eigen::Matrix<double, m, m> Quu_inverse = eigenSolver.eigenvectors() * eigenValues.asDiagonal() * eigenSolver.eigenvectors().transpose();
//  std::cout << "\t->\t" << Quu;

  //  Quu = Cuu + Fu.transpose() * (V + mu * V_t::Identity()) * Fu;
  //  Qxu = (Cxu.transpose() + Fu.transpose() * (V + mu * V_t::Identity()) * Fx).transpose();

  //  std::cout << "Q:\n" << Q << "\n";
  //  std::cout << "q:\n" << q << "\n";
  //  std::cout << "V:\n" << V << "\n";

  //  // Quu is not PD
  //  Eigen::LLT<Eigen::Matrix<double, m, m>> llt(Quu);
  //  if (llt.info() == Eigen::NumericalIssue)
  //  {
  //    std::cerr << "\033[31;1m"
  //              << "Quu was not PD. mu: " << mu << ", Quu:\n"
  //              << Quu << "\nCuu:\n"
  //              << Cuu
  //              << "\nV:\n"
  //              << V << "\nCuu + Fu^T V Fu:\n"
  //              << (Cuu + Fu.transpose() * V * Fu)
  //              << "\nmu * I:\n"
  //              << (mu * V_t::Identity()) << "\n Fu (V + mu * I):\n"
  //              << (Fu.transpose() * (V + mu * V_t::Identity()) * Fu) << "\033[0m" << "\n\n";
  //    return std::nullopt;
  //  }

  //  std::cout << "\nQuu: " << getuu<n, m>(Q) << ", Qxu: " << getxu<n, m>(Q).transpose() << ", qu: " << getu<n,
  //  m>(q).transpose() << ", F:\n" << dynamics.F.transpose() << ", f: " << dynamics.f.transpose() << "\n"; std::cout <<
  //  "c: " << costs.c.transpose() << ", V:\n" << V.transpose() << ", v: " << v.transpose() << "\n";
//  Eigen::Matrix<double, m, m> Quu_inverse = Eigen::Matrix<double, m, m>(getuu<n, m>(Q)).inverse();
//  std::cout << ", Quu_inverse: " << Quu_inverse << "\n";

  K_t K = -Quu_inverse * getxu<n, m>(Q).transpose();
  k_t k = -Quu_inverse * getu<n, m>(q);

  //  std::cout << "Quu: " << Quu << ", Qxu: " << getxu<n, m>(Q).transpose() << "Quu_inverse: " << Quu_inverse << ", qu:
  //  " << getu<n, m>(q) << "\n";

  dV(0) += (k.transpose() * getu<n, m>(q))(0);
  dV(1) += (k.transpose() * getuu<n, m>(Q) * k)(0);
  V = getxx<n, m>(Q) + K.transpose() * getuu<n, m>(Q) * K + 2 * getxu<n, m>(Q) * K;
  v = getx<n, m>(q) + K.transpose() * getuu<n, m>(Q) * k + K.transpose() * getu<n, m>(q) + getxu<n, m>(Q) * k;

  LQRStepResult out{ K, k };
  return out;
}

template <int n, int m>
double ILQRController<n, m>::forwardPropogate(std::vector<State_t> &states,
                                              const std::vector<Control_t> &controls) const
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

  return cost;
}

template <int n, int m>
std::vector<typename LQRController<n, m>::Dynamics> ILQRController<n, m>::approximateDynamics(
    const std::vector<State_t> &states, const std::vector<Control_t> &controls) const
{
  const int num_controls = controls.size();

  std::vector<typename LQR::Dynamics> F_array;
  F_array.reserve(num_controls);

  auto wrapped = [=](const Full_t &full) -> State_t {
    State_t x = full.head(n);
    Control_t u = full.tail(m);
    auto res = dynamics_function_(x, u);
    //    printf("x: (%8e, %8e), u: %8e \t -> \t (%8e, %8e)\n", x(0), x(1), u(0), res(0), res(1));
    return res;
  };

  constexpr double epsilon = 1e-6;

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
  constexpr double epsilon = 1e-6;

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

template <int n, int m>
void ILQRController<n, m>::printState(const std::vector<State_t> &states, const std::vector<Control_t> &controls) const
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
template <int n, int m>
void ILQRController<n, m>::printState(const std::vector<State_t> &states, const std::vector<Control_t> &controls,
                                      const std::vector<Control_t> control_deltas) const
{
  printState(states, controls);
  std::cout << ">>>> CONTROL DELTAS:\n";
  for (const auto &u : control_deltas)
  {
    std::cout << "(" << u.transpose() << ") ";
  }
  std::cout << "\n\n";
}

template <int n, int m>
void ILQRController<n, m>::increaseMu()
{
  delta = std::max(delta_zero, delta * delta);
  mu = std::max(mu_min, mu * delta);
}

template <int n, int m>
void ILQRController<n, m>::decreaseMu()
{
  delta = std::min(1 / delta_zero, delta / delta_zero);
  mu = mu * delta > mu_min ? mu * delta : 0.0;
}

}  // namespace controllers

#endif  // CONTROLS_TEST_ILQR_TPP
