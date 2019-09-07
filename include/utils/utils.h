#ifndef CONTROLS_TEST_UTILS_H
#define CONTROLS_TEST_UTILS_H

#include <Eigen/Core>

namespace utils
{
template <int n, int m, typename Function>
[[nodiscard]] auto createFullFunction(Function f)
{
  using State_t = Eigen::Matrix<double, n, 1>;
  using Control_t = Eigen::Matrix<double, m, 1>;
  using Full_t = Eigen::Matrix<double, n+m, 1>;

  using FunctionReturnType = typename std::invoke_result_t<Function, const State_t&, const Control_t&>;
  auto full_function = [=](const Full_t& full) -> FunctionReturnType {
    State_t state = full.template head<n>();
    Control_t control = full.template tail<m>();

    return f(state, control);
  };

  return full_function;
}

template <int n, int m>
[[nodiscard]] Eigen::Matrix<double, n+m, 1> createFullState(const Eigen::Matrix<double, n, 1>& x, const Eigen::Matrix<double, m, 1>& u)
{
  Eigen::Matrix<double, n+m, 1> full_state;
  full_state << x, u;
  return full_state;
}
}

#endif //CONTROLS_TEST_UTILS_H
