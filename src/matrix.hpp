#pragma once
#include <vector>
#include <cstdint>

using GF2Matrix = std::vector<std::vector<bool>>;
using GF2Vector = std::vector<bool>;

size_t rows(const GF2Matrix& A);
size_t cols(const GF2Matrix& A);

GF2Matrix gf2Init(size_t rows, size_t cols);
GF2Matrix gf2Identity(size_t size);
GF2Matrix gf2Rotl(size_t size, int amount);
GF2Matrix gf2Rotr(size_t size, int amount);
GF2Matrix gf2Shiftl(size_t size, int amount);
GF2Matrix gf2Xor(const GF2Matrix& A, const GF2Matrix& B);
GF2Matrix gf2Multiply(const GF2Matrix& A, const GF2Matrix& B);
GF2Matrix gf2Echelon(const GF2Matrix& A, GF2Matrix& transformMat, int& rank, std::vector<int>& pivots);
GF2Matrix gf2Inverse(const GF2Matrix& A);
GF2Matrix gf2NullBasis(const GF2Matrix& A);
GF2Matrix gf2NullSpace(const GF2Matrix& A);

std::vector<uint64_t> gf2Pack(const GF2Matrix& A);
uint64_t packVector(const GF2Vector& A);

void gf2Print(const GF2Matrix& A);