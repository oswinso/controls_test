#ifndef CONTROLS_TEST_LQR_H
#define CONTROLS_TEST_LQR_H

#include <Eigen/Core>

namespace controllers
{
class LQRController
{
 public:
  LQRController() noexcept;
  [[nodiscard]] solve();
 private:
};
}

#endif  // CONTROLS_TEST_LQR_H
