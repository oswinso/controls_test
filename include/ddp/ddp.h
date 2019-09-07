#ifndef CONTROLS_TEST_DDP_H
#define CONTROLS_TEST_DDP_H

#include <lqr/lqr.h>
#include <numerics/approximation.h>
#include <Eigen/Core>
#include <functional>

namespace controllers
{
template <int n, int m>
class DDPController
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

  using DynamicsFunction = std::function<State_t(const State_t& x, const Control_t& u)>;
  using CostFunction = std::function<double(const State_t& x, const Control_t& u)>;
  using FinalCostFunction = std::function<double(const State_t& x)>;

  struct Dynamics
  {
    numerics::HessianTensor<n + m> F_hessian;
    F_t F;
  };

  struct Costs
  {
    C_t C;
    c_t c;
  };

  struct FinalCosts
  {
    V_t V_final;
    v_t v_final;
  };

  DDPController(const DynamicsFunction& dynamics_function, const CostFunction& cost_function,
                const FinalCostFunction& final_cost_function) noexcept;

  [[nodiscard]] std::vector<Control_t> solve(const State_t& x0, int timesteps, int iterations);

private:
  DynamicsFunction dynamics_function_;
  CostFunction cost_function_;
  FinalCostFunction final_cost_function_;

  double mu = 1.0;
  double mu_min = 1e-6;
  double delta_zero = 2;
  double delta = 1.0;

  struct StepResult
  {
    K_t K;
    k_t k;
  };

  Eigen::Vector2d dV{ 0.0, 0.0 };

  void increaseMu();
  void decreaseMu();

  double forwardPropogate(std::vector<State_t>& states, const std::vector<Control_t>& controls) const;

  std::vector<Dynamics> approximateDynamics(const std::vector<State_t>& states,
                                            const std::vector<Control_t>& controls) const;

  std::pair<std::vector<Costs>, FinalCosts> approximateCosts(const std::vector<State_t>& states,
                                                             const std::vector<Control_t>& controls) const;

  std::vector<StepResult> backwardsPass(int timesteps, const std::vector<Dynamics>& dynamics,
                                        const std::vector<Costs>& costs, const FinalCosts& final_cost);
  std::optional<StepResult> backwardsStep(V_t& V, v_t& v, const Costs& costs, const Dynamics& dynamics);
  Quu_t getRegularizedInverse(const Eigen::Ref<Quu_t>& Quu) const;

  void printState(const std::vector<State_t>& states, const std::vector<Control_t>& controls) const;
};
}  // namespace controllers
#include "ddp.tpp"

#endif  // CONTROLS_TEST_DDP_H
