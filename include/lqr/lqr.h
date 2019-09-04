#pragma once
#ifndef CONTROLS_TEST_LQR_H
#define CONTROLS_TEST_LQR_H

#include <Eigen/Core>

namespace controllers
{
template <int n, int m>
class LQRController
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

  using ControlArray_t = Eigen::Matrix<double, m, Eigen::Dynamic>;

  struct Dynamics
  {
    F_t F;
    f_t f;

    [[nodiscard]] State_t propogate(const State_t& x, const Control_t& u) const;
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

  struct LQRStepResult
  {
    K_t K;
    k_t k;
  };

  LQRController() noexcept;
  LQRController(const Dynamics& dynamics, const Costs& costs, const FinalCosts& final_costs) noexcept;

  [[nodiscard]] ControlArray_t solve(const State_t& x0, int timesteps) const;
  [[nodiscard]] std::vector<LQRStepResult> solve(int timesteps, const std::vector<Dynamics>& dynamics,
                                             const std::vector<Costs>& costs, const FinalCosts& final_costs) const;
  void setDynamics(const Dynamics& dynamics) noexcept;
  void setCosts(const Costs& dynamics) noexcept;

private:
  Dynamics dynamics_;
  Costs costs_;
  FinalCosts final_costs_;

  static LQRStepResult lqrStep(V_t& V, v_t& v, const Costs& costs, const Dynamics& dynamics);

  [[nodiscard]] static auto getxx(const Eigen::Matrix<double, n + m, n + m>& matrix);
  [[nodiscard]] static auto getxu(const Eigen::Matrix<double, n + m, n + m>& matrix);
  [[nodiscard]] static auto getuu(const Eigen::Matrix<double, n + m, n + m>& matrix);

  [[nodiscard]] static auto getx(const Eigen::Matrix<double, n + m, 1>& matrix);
  [[nodiscard]] static auto getu(const Eigen::Matrix<double, n + m, 1>& matrix);
};
}  // namespace controllers
#include "lqr.tpp"

#endif  // CONTROLS_TEST_LQR_H
