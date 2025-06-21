#include "pokemon_cuda.h"
#include "generatorReversal.cuh"
#include <iostream>

__device__ void findGeneratorSeeds(uint64_t seed, uint32_t* seedCount, uint64_t outputBuffer[GENERATOR_OUTPUT_BUFFER_SIZE], const uint64_t slices[256]) {
    const unsigned int x = ((blockIdx.x >> 0) & 31) * 8 + threadIdx.x;
    const unsigned int y = ((blockIdx.x >> 5) & 31) * 8 + threadIdx.y;
    const unsigned int z = ((blockIdx.x >> 10) & 31) * 8 + threadIdx.z;

    const uint64_t x1_slice = slices[x] | (slices[y] << 1) | (slices[z] << 2);
    const uint64_t x0_slice = x0_from_x1(x1_slice) & 0x3838383838383838ULL;
    const uint64_t sub = seed - x0_slice - x1_slice;

    const uint64_t base_x1_slice_3 = sub & 0x808080808080808ULL;
    const uint64_t sub_carry = sub - 0xc0c0c0c0c0c0c0c0ULL;
    const uint64_t changed = (sub_carry ^ sub) & 0x808080808080808ULL;

    for (uint64_t i0 = 0; i0 <= ((changed >> 11) & 1); i0++) {
        uint64_t part_0 = i0 << 11;
        for (uint64_t i1 = 0; i1 <= ((changed >> 19) & 1); i1++) {
            uint64_t part_1 = part_0 | (i1 << 19);
            for (uint64_t i2 = 0; i2 <= ((changed >> 27) & 1); i2++) {
                uint64_t part_2 = part_1 | (i2 << 27);
                for (uint64_t i3 = 0; i3 <= ((changed >> 35) & 1); i3++) {
                    uint64_t part_3 = part_2 | (i3 << 35);
                    for (uint64_t i4 = 0; i4 <= ((changed >> 43) & 1); i4++) {
                        uint64_t part_4 = part_3 | (i4 << 43);
                        for (uint64_t i5 = 0; i5 <= ((changed >> 51) & 1); i5++) {
                            uint64_t part_5 = part_4 | (i5 << 51);
                            for (uint64_t i6 = 0; i6 <= ((changed >> 59) & 1); i6++) {
                                uint64_t part_6 = part_5 | (i6 << 59);
                                uint64_t x1_slice_3 = base_x1_slice_3 ^ part_6;
                                uint64_t x0_slice_3 = x0_from_x1(x1_slice_3) & 0x4040404040404040ULL;

                                uint64_t x0_slice_ = x0_slice | x0_slice_3;
                                uint64_t x1_slice_ = x1_slice | x1_slice_3;

                                uint64_t x1_slice_4 = (seed - x0_slice_ - x1_slice_) & 0x1010101010101010ULL;
                                uint64_t x0_slice_4 = x0_from_x1(x1_slice_4) & 0x8080808080808080ULL;

                                x0_slice_ |= x0_slice_4;
                                x1_slice_ |= x1_slice_4;

                                uint64_t x1_slice_567 = (seed - x0_slice_ - x1_slice_) & 0xe0e0e0e0e0e0e0e0ULL;
                                uint64_t x0_slice_567 = x0_from_x1(x1_slice_567) & 0x707070707070707ULL;

                                uint64_t x0 = x0_slice_ | x0_slice_567;
                                uint64_t x1 = x1_slice_ | x1_slice_567;

                                if ((x0 + x1) == seed) {
                                    uint64_t seed = seed_from_x1(x1);
                                    outputBuffer[atomicInc(seedCount, 1)] = seed;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void kernel_findGeneratorSeedsSingle(uint64_t seed, uint32_t* numFoundSeeds, uint64_t outputBuffer[GENERATOR_OUTPUT_BUFFER_SIZE], const uint64_t slices[256]) {
    findGeneratorSeeds(seed, numFoundSeeds, outputBuffer, slices);
}

int genSeedWrapper(uint64_t seed, uint64_t* outputBuffer) {
    uint64_t slices[256] = {0};
    for (uint64_t i = 0; i < 256; i++) {
        for (uint64_t j = 0; j < 8; j++) {
            if ((i >> j) & 1) {
                slices[i] |= (1ull << (j * 8));
            }
        }
    }
    uint32_t* cnt;
    cudaMalloc(&cnt, sizeof(uint32_t));
    auto result = cudaMemset(cnt, 0, sizeof(uint32_t));
    uint64_t* deviceOutputBuffer;
    cudaMalloc(&deviceOutputBuffer, OUTPUT_BUFFER_SIZE * sizeof(uint64_t));
    uint64_t* deviceSlices;
    cudaMalloc(&deviceSlices, 256 * sizeof(uint64_t));
    result = cudaMemcpy(deviceSlices, slices, 256 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    dim3 block(8, 8, 8);
    kernel_findGeneratorSeedsSingle<<<32768, block>>>(seed, cnt, deviceOutputBuffer, deviceSlices);
    uint32_t numSeeds;
    cudaMemcpy(&numSeeds, cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(outputBuffer, deviceOutputBuffer, numSeeds * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(cnt);
    cudaFree(deviceSlices);
    cudaFree(deviceOutputBuffer);
    return numSeeds;
}