#ifndef CONTROLS_TEST_APPROXIMATION_H
#define CONTROLS_TEST_APPROXIMATION_H

#include <Eigen/Core>

namespace numerics
{
/**
 * Linearizes a function f: Rm -> Rn
 * @tparam n dimension of the output of f
 * @tparam m dimension of the input of f
 * @tparam Function
 * @param f function to linearize
 * @param x point to linearize around
 * @param epsilon
 * @return Jacobian of f
 */
template <int n, int m, typename Function>
[[nodiscard]] Eigen::Matrix<double, n, m> linearize(Function f, const Eigen::Matrix<double, m, 1>& x,
                                                    double epsilon = 1e-9) {
  using Input_t = Eigen::Matrix<double, m, 1>;
  using Output_t = Eigen::Matrix<double, n, 1>;
  using Jacobian_t = Eigen::Matrix<double, n, m>;

  Jacobian_t jacobian;

  Input_t delta = Input_t::Zero();
  for (int i = 0; i < m; i++)
  {
    delta(i) = epsilon;
    Output_t positive = f(x + delta);
    Output_t negative = f(x - delta);
    jacobian.col(i) = (positive - negative) / (2 * epsilon);
    delta(i) = 0.0;
  }

  return jacobian;
}

/**
 * Linearizes a function f: Rm -> R
 * @tparam m dimension of the input of f
 * @tparam Function
 * @param f function to linearize
 * @param x point to linearize around
 * @param epsilon
 * @return Jacobian of f
 */
template <int m, typename Function>
[[nodiscard]] Eigen::Matrix<double, 1, m> linearizeScalar(Function f, const Eigen::Matrix<double, m, 1>& x,
                                                    double epsilon = 1e-9) {
  using Input_t = Eigen::Matrix<double, m, 1>;
  using Jacobian_t = Eigen::Matrix<double, 1, m>;

  Jacobian_t jacobian;

  Input_t delta = Input_t::Zero();
  for (int i = 0; i < m; i++)
  {
    delta(i) = epsilon;
    double positive = f(x + delta);
    double negative = f(x - delta);
    jacobian(i) = (positive - negative) / (2 * epsilon);
    delta(i) = 0.0;
  }

  return jacobian;
}

/**
 * Quadratizes a function f: Rm -> Rn
 * @tparam n dimension of output
 * @tparam m dimension of input
 * @tparam Function
 * @param f function to quadratize
 * @param x point to quadratize around
 * @param epsilon
 * @return Hessian of
 */
template <int n, typename Function>
[[nodiscard]] Eigen::Matrix<double, n, n> quadratize(Function f, const Eigen::Matrix<double, n, 1>& x,
                                                     double epsilon = 1e-3) {
  using Input_t = Eigen::Matrix<double, n, 1>;
  using Hessian_t = Eigen::Matrix<double, n, n>;

  Hessian_t hessian;

  Input_t d1 = Input_t::Zero();
  Input_t d2 = Input_t::Zero();
  for (int i = 0; i < n; i++)
  {
    d1(i) = epsilon;
    for (int j = 0; j < n; j++)
    {
      d2(j) = epsilon;
      hessian(i, j) = (f(x + d1 + d2) - f(x+d1) - f(x+d2) + f(x)) / (epsilon * epsilon);
//      if (i == j)
//      {
//        Input_t arg_1 = x + 2 * d1;
//        Input_t arg_2 = x + d1;
//        Input_t arg_3 = x - d1;
//        Input_t arg_4 = x - 2 * d1;
//        hessian(i, j) = (-f(arg_1) + 16 * f(arg_2) - 30 * f(x) + 16 * f(arg_3) - f(arg_4)) / (12 * epsilon * epsilon);
//      }
//      else
//      {
//        Input_t arg_1 = x + d1 + d2;
//        Input_t arg_2 = x + d1 - d2;
//        Input_t arg_3 = x - d1 + d2;
//        Input_t arg_4 = x - d1 - d1;
//        hessian(i, j) = (f(arg_1) - f(arg_2) - f(arg_3) + f(arg_4)) / (4 * epsilon * epsilon);
//      }
      d2(j) = 0.0;
    }
    d1(i) = 0.0;
  }
Hessian_t symmetric_hessian = 0.5 * (hessian + hessian.transpose());
//  std::cout << "Hessian:\n" << symmetric_hessian << "\n";
  return symmetric_hessian;
}

}  // namespace numerics

#endif  // CONTROLS_TEST_APPROXIMATION_H
