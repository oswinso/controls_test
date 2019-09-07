#ifndef CONTROLS_TEST_COST_BLOCKS_H
#define CONTROLS_TEST_COST_BLOCKS_H

#include <Eigen/Core>

template <int n, int m>
[[nodiscard]] auto getxx(const Eigen::Matrix<double, n + m, n + m> &matrix)
{
  return matrix.template block<n, n>(0, 0);
}

template <int n, int m>
[[nodiscard]] auto getxu(const Eigen::Matrix<double, n + m, n + m> &matrix)
{
  return matrix.template block<n, m>(0, n);
}

template <int n, int m>
[[nodiscard]] auto getuu(const Eigen::Matrix<double, n + m, n + m> &matrix)
{
  return matrix.template block<m, m>(n, n);
}

template <int n, int m>
[[nodiscard]] auto getx(const Eigen::Matrix<double, n + m, 1> &matrix)
{
  return matrix.template block<n, 1>(0, 0);
}

template <int n, int m>
[[nodiscard]] auto getu(const Eigen::Matrix<double, n + m, 1> &matrix)
{
  return matrix.template block<m, 1>(n, 0);
}

#endif //CONTROLS_TEST_COST_BLOCKS_H
