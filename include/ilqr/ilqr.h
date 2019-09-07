#ifndef CONTROLS_TEST_ILQR_H
#define CONTROLS_TEST_ILQR_H

#include <lqr/lqr.h>
#include <Eigen/Core>
#include <functional>

namespace controllers
{
template <int n, int m>
class ILQRController
{
public:
  using F_t = Eigen::Matrix<double, n, n + m>;
  using f_t = Eigen::Matrix<double, n, 1>;
  using C_t = Eigen::Matrix<double, n + m, n + m>;
  using c_t = Eigen::Matrix<double, n + m, 1>;

  using Q_t = Eigen::Matrix<double, n + m, n + m>;
  using Quu_t = Eigen::Matrix<double, m, m>;
  using Qux_t = Eigen::Matrix<double, m, n>;
  using q_t = Eigen::Matrix<double, n + m, 1>;

  using V_t = Eigen::Matrix<double, n, n>;
  using v_t = Eigen::Matrix<double, n, 1>;

  using K_t = Eigen::Matrix<double, m, n>;
  using k_t = Eigen::Matrix<double, m, 1>;

  using State_t = Eigen::Matrix<double, n, 1>;
  using Control_t = Eigen::Matrix<double, m, 1>;
  using Full_t = Eigen::Matrix<double, n + m, 1>;

  struct ApproximateCosts
  {
    std::vector<C_t> Cs;
    std::vector<c_t> cs;
    V_t V_final;
    v_t v_final;
  };

  using DynamicsFunction = std::function<State_t(const State_t& x, const Control_t& u)>;
  using CostFunction = std::function<double(const State_t& x, const Control_t& u)>;
  using FinalCostFunction = std::function<double(const State_t& x)>;

  using LQR = LQRController<n, m>;
  using Costs = typename LQR::Costs;
  using Dynamics = typename LQR::Dynamics;
  using FinalCosts = typename LQR::FinalCosts;

  using LQRStepResult = typename LQR::LQRStepResult;
  using StepResult = std::optional<typename LQR::LQRStepResult>;

  ILQRController(const DynamicsFunction& dynamics_function, const CostFunction& cost_function,
                 const FinalCostFunction& final_cost_function) noexcept;

  [[nodiscard]] std::vector<Control_t> solve(const State_t& x0, int timesteps);

private:
  DynamicsFunction dynamics_function_;
  CostFunction cost_function_;
  FinalCostFunction final_cost_function_;
  LQR lqr_controller_;

  double mu = 1.0;
  double mu_min = 1e-6;
  double delta_zero = 2;
  double delta = 1.0;

  Eigen::Vector2d dV {0.0, 0.0};

  double forwardPropogate(std::vector<State_t>& states, const std::vector<Control_t>& controls) const;

  std::vector<typename LQR::Dynamics> approximateDynamics(const std::vector<State_t>& states,
                                                          const std::vector<Control_t>& controls) const;

  void printState(const std::vector<State_t>& states, const std::vector<Control_t>& controls) const;
  void printState(const std::vector<State_t>& states, const std::vector<Control_t>& controls,
                  const std::vector<Control_t> control_deltas) const;

  std::vector<LQRStepResult> backwardsPass(int timesteps, const std::vector<Dynamics>& dynamics,
                                        const std::vector<Costs>& costs, const FinalCosts& final_costs);
  StepResult backwardsStep(V_t& V, v_t& v, const Costs& costs, const Dynamics& dynamics);

  void increaseMu();
  void decreaseMu();

  std::pair<std::vector<typename LQR::Costs>, typename LQR::FinalCosts>
  approximateCost(const std::vector<State_t>& states, const std::vector<Control_t>& controls) const;
};
}  // namespace controllers
#include "ilqr.tpp"

#endif  // CONTROLS_TEST_ILQR_H
