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

  using ControlArray_t = Eigen::Matrix<double, m, Eigen::Dynamic>;
  using StateArray_t = Eigen::Matrix<double, n, Eigen::Dynamic>;

  using FArray_t = Eigen::Matrix<double, n, Eigen::Dynamic>;

  using CArray_t = Eigen::Matrix<double, n + m, Eigen::Dynamic>;
  using cArray_t = Eigen::Matrix<double, n + m, Eigen::Dynamic>;

  struct ApproximateCosts
  {
    CArray_t Cs;
    cArray_t cs;
  };

  using DynamicsFunction = std::function<State_t(const State_t& x, const Control_t& u)>;
  using CostFunction = std::function<double(const State_t& x, const Control_t& u)>;

  ILQRController(const DynamicsFunction& dynamics_function, const CostFunction& cost_function) noexcept;

  [[nodiscard]] ControlArray_t solve(const State_t& x0, int timesteps) const;

private:
  DynamicsFunction dynamics_function_;
  CostFunction cost_function_;
  LQRController<n, m> lqr_controller_;

  void forwardPropogate(StateArray_t& states, const ControlArray_t& controls) const;

  FArray_t approximateDynamics(const StateArray_t& states, const ControlArray_t& controls) const;
  ApproximateCosts approximateCost(const StateArray_t& states, const ControlArray_t& controls) const;
};
}  // namespace controllers
#include "ilqr.tpp"

#endif  // CONTROLS_TEST_ILQR_H
