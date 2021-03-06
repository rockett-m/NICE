// The MIT License (MIT)
//
// Copyright (c) 2016 Northeastern University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


// This file tests the cpu_operations.cc LogicalAnd() function by calling the
// Function on two randomly initialized matrices and checking the results.
// Behavior with oddly shaped matrices (oddshape tests) is also tested with
// Matrices of size (3, 4) and (2, 3). The data type of matrix passed to the
// Function is not tested  because Eigen contains its own internal asserts

#include <stdio.h>
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"

// This is a test fixture class containing NICE matrices
class MyTest : public ::testing::Test {  // Inherits from testing::Test
 public:
  Nice::Matrix<bool> _matrix_nice_1;  // First boolean Matrix
  Nice::Matrix<bool> _matrix_nice_2;  // Second boolean Matrix
  Nice::Matrix<bool> _logical_and;  // Resulting Matrix

  // Calls the LogicalAnd function on the Matrices
  void LogicalAnd() {
    // Apply logic to matrixNice
    _logical_and = Nice::CpuOperations<bool>::LogicalAnd(_matrix_nice_1,
                                                         _matrix_nice_2);
  }
};

// This test checks the functionality of LogicalAnd by creating two matrices
// And comparing the result of the function with the and of each element
TEST_F(MyTest, LogicalAndFunctionality) {
  this->_matrix_nice_1.setRandom(3, 3);  // Random bool values
  this->_matrix_nice_2.setRandom(3, 3);
  this->_logical_and.setZero(3, 3);  // Set the _logical_and to zero
  this->LogicalAnd();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // Check equality for the and of each element
      EXPECT_EQ(((this->_matrix_nice_1(i, j)) && (this->_matrix_nice_2(i, j))),
                (this->_logical_and(i, j)));
    }
  }
}

// This test checks the functionality of LogicalAnd by creating two matrices
// Of different size and asserting a death when the function is called
TEST_F(MyTest, LogicalAndTestWrongSize) {
  this->_matrix_nice_1.setRandom(3, 4);  // Random bool values
  this->_matrix_nice_2.setRandom(2, 3);
  this->_logical_and.setZero(3, 3);
  ASSERT_DEATH({this->LogicalAnd();}, ".*");  // Expect failure
  // ".*" denotes any failure message
}

