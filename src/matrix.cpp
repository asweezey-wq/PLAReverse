#include "matrix.hpp"
#include "xoroshiro.hpp"

#include <cstdint>
#include <algorithm>
#include <vector>
#include <iostream>
#include <assert.h>

int rows(const GF2Matrix& A) {
    return A.size();
}

int cols(const GF2Matrix& A) {
    return A.size() == 0 ? 0 : A[0].size();
}

GF2Matrix gf2Init(int rows, int cols) {
    return GF2Matrix(rows, GF2Vector(cols, 0));
}

GF2Matrix gf2Identity(int size) {
    GF2Matrix mat = gf2Init(size, size);
    for (int i = 0; i < size; i++) {
        mat[i][i] = 1;
    }
    return mat;
}

GF2Matrix gf2Multiply(const GF2Matrix& A, const GF2Matrix& B) {
    assert(cols(A) == rows(B));
    GF2Matrix result = gf2Init(rows(A), cols(B));
    for (int i = 0; i < rows(result); i++) {
        for (int j = 0; j < cols(result); j++) {
            bool value = 0;
            for (int k = 0; k < cols(A); k++) {
                value ^= (A[i][k] & B[k][j]);
            }
            result[i][j] = value;
        }
    }
    return result;
}

GF2Matrix gf2Echelon(const GF2Matrix& A, GF2Matrix& transformMat, int& rank, std::vector<int>& pivots) {
    int m = rows(A);
    int n = cols(A);
    GF2Matrix echelonMat = GF2Matrix(A);
    transformMat = gf2Identity(m);
    rank = 0;
    pivots.clear();
    for (int j = 0; j < n; j++) {
        for (int i = rank; i < m; i++) {
            if (echelonMat[i][j]) {
                for (int k = 0; k < m; k++) {
                    if (k != i && echelonMat[k][j]) {
                        for (int x = 0; x < m; x++) {
                            if (x < n) {
                                echelonMat[k][x] = echelonMat[k][x] ^ echelonMat[i][x];
                            }
                            transformMat[k][x] = transformMat[k][x] ^ transformMat[i][x];
                        }
                    }
                }
                std::swap(echelonMat[i], echelonMat[rank]);
                std::swap(transformMat[i], transformMat[rank]);
                pivots.push_back(j);
                rank += 1;
                break;
            }
        }
    }
    return echelonMat;
}

GF2Matrix gf2Inverse(const GF2Matrix& A) {
    GF2Matrix transformMat;
    int rank = 0;
    std::vector<int> pivots;
    GF2Matrix echelonMat = gf2Echelon(A, transformMat, rank, pivots);
    for (int i = rank - 1; i >= 0; i--) {
        int col = pivots[i];
        std::swap(transformMat[i], transformMat[col]);
    }
    int n = cols(transformMat);
    transformMat.resize(cols(A), GF2Vector(n, 0));
    return transformMat;
}

GF2Matrix gf2NullBasis(const GF2Matrix& A) {
    GF2Matrix invMat = gf2Inverse(A);
    GF2Matrix basis = gf2Multiply(A, invMat);
    for (int i = 0; i < rows(basis); i++) {
        basis[i][i] = !basis[i][i];
    }
    GF2Matrix transformMat;
    int rank;
    std::vector<int> pivots;
    GF2Matrix echelon = gf2Echelon(basis, transformMat, rank, pivots);
    echelon.resize(rank);
    return echelon;
}

GF2Matrix gf2NullSpace(const GF2Matrix& A) {
    GF2Matrix basis = gf2NullBasis(A);
    GF2Matrix space = gf2Init(1 << rows(basis), cols(basis));
    for (int i = 0; i < rows(space); i++) {
        for (int j = 0; j < rows(basis); j++) {
            if ((i >> j) & 1) {
                for (int k = 0; k < cols(basis); k++) {
                    space[i][k] = space[i][k] ^ basis[j][k];
                }
            }
        }
    }
    return space;
}

void gf2Print(const GF2Matrix& A) {
    for (int i = 0; i < rows(A); i++) {
        for (int j = 0; j < cols(A); j++) {
            bool b = A[i][j];
            printf("%d ", b);
        }
        printf("\n");
    }
}

std::vector<uint64_t> gf2Pack(const GF2Matrix& A) {
    std::vector<uint64_t> result;
    for (const auto& v : A) {
        result.push_back(packVector(v));
    }
    return result;
}

uint64_t packVector(const GF2Vector& A) {
    assert(A.size() <= 64);
    uint64_t result = 0;
    for (int i = 0; i < A.size(); i++) {
        result |= ((uint64_t)A[i] << i);
    }
    return result;
}