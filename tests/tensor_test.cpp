#include <gtest/gtest.h>

#include "Tensor.h"

TEST(TensorTest, NumelComputesProductOfShape) {
  titan::Tensor t({2, 3, 4});
  EXPECT_EQ(t.numel(), 24u);
}

TEST(TensorTest, DefaultConstructedTensorIsEmpty) {
  titan::Tensor t;
  EXPECT_TRUE(t.shape().empty());
  EXPECT_EQ(t.numel(), 1u);
}
