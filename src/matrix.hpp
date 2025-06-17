#pragma once
#include <vector>
#include <cstdint>

using GF2Matrix = std::vector<std::vector<bool>>;
using GF2Vector = std::vector<bool>;

int rows(const GF2Matrix& A);
int cols(const GF2Matrix& A);

GF2Matrix gf2Init(int rows, int cols);
GF2Matrix gf2Identity(int size);
GF2Matrix gf2Multiply(const GF2Matrix& A, const GF2Matrix& B);
GF2Matrix gf2Echelon(const GF2Matrix& A, GF2Matrix& transformMat, int& rank, std::vector<int>& pivots);
GF2Matrix gf2Inverse(const GF2Matrix& A);
GF2Matrix gf2NullBasis(const GF2Matrix& A);
GF2Matrix gf2NullSpace(const GF2Matrix& A);

std::vector<uint64_t> gf2Pack(const GF2Matrix& A);
uint64_t packVector(const GF2Vector& A);

void gf2Print(const GF2Matrix& A);