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
    using Cxx_t = Eigen::Matrix<double, n, n>;
    using Cxu_t = Eigen::Matrix<double, n, m>;
    using Cuu_t = Eigen::Matrix<double, m, m>;

    using cx_t = Eigen::Matrix<double, n, 1>;
    using cu_t = Eigen::Matrix<double, m, 1>;

    C_t C;
    c_t c;
  };

  LQRController() noexcept = default;
  LQRController(const Dynamics& dynamics, const Costs& costs) noexcept;

  [[nodiscard]] ControlArray_t solve(const State_t& x0, int timesteps) const;
  void setDynamics(const Dynamics& dynamics) noexcept;
  void setCosts(const Costs& dynamics) noexcept;

private:
  Dynamics dynamics_;
  Costs costs_;

  [[nodiscard]] auto getxx(const Eigen::Matrix<double, n+m, n+m>& matrix) const;
  [[nodiscard]] auto getxu(const Eigen::Matrix<double, n+m, n+m>& matrix) const;
  [[nodiscard]] auto getuu(const Eigen::Matrix<double, n+m, n+m>& matrix) const;

  [[nodiscard]] auto getx(const Eigen::Matrix<double, n+m, 1>& matrix) const;
  [[nodiscard]] auto getu(const Eigen::Matrix<double, n+m, 1>& matrix) const;
};
}  // namespace controllers
#include "lqr.tpp"

#endif  // CONTROLS_TEST_LQR_H
