#pragma once
#include "matrix.hpp"
#include <string>
#include <vector>

GF2Matrix computeIVMatrix(int numShinyRolls);
GF2Vector computeIVConst(int numShinyRolls);

std::vector<uint64_t> reverseIVSet(int numShinyRolls, uint8_t ivs1[6], uint8_t ivs2[6]);