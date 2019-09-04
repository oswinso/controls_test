#ifndef CONTROLS_TEST_ILQR_H
#define CONTROLS_TEST_ILQR_H

#include <functional>
#include <lqr/lqr.h>
#include <Eigen/Core>

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

  ILQRController(const DynamicsFunction& dynamics_function, const CostFunction& cost_function, const FinalCostFunction& final_cost_function) noexcept;

  [[nodiscard]] std::vector<Control_t> solve(const State_t& x0, int timesteps) const;

private:
  using LQR = LQRController<n, m>;
  DynamicsFunction dynamics_function_;
  CostFunction cost_function_;
  FinalCostFunction final_cost_function_;
  LQR lqr_controller_;

  void forwardPropogate(std::vector<State_t>& states, const std::vector<Control_t>& controls) const;

  std::vector<typename LQR::Dynamics> approximateDynamics(const std::vector<State_t>& states, const std::vector<Control_t>& controls) const;

  void printState(const std::vector<State_t>& states, const std::vector<Control_t>& controls) const;
  void printState(const std::vector<State_t>& states, const std::vector<Control_t>& controls, const std::vector<Control_t> control_deltas) const;

  std::pair<std::vector<typename LQR::Costs>, typename LQR::FinalCosts> approximateCost(const std::vector<State_t>& states, const std::vector<Control_t>& controls) const;
};
}  // namespace controllers
#include "ilqr.tpp"

#endif  // CONTROLS_TEST_ILQR_H
