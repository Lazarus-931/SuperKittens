//
//  accumilate.h
//  SuperKittens
//
//  Lightweight accumulator tiles for GEMM reference kernels.
//  File name kept as-is to avoid churn while the GEMM path is still settling.
//

#ifndef SUPERKITTENS_GEMM_ACCUMILATE_H
#define SUPERKITTENS_GEMM_ACCUMILATE_H

namespace meow {
namespace gemm {

template <int ROWS, int COLS>
struct AccumulatorTile {
    float data[ROWS][COLS];

    METAL_FUNC void clear() {
        for (int r = 0; r < ROWS; ++r)
            for (int c = 0; c < COLS; ++c)
                data[r][c] = 0.0f;
    }

    METAL_FUNC void add(int row, int col, float value) {
        data[row][col] += value;
    }
};

} // namespace gemm
} // namespace meow

#endif // SUPERKITTENS_GEMM_ACCUMILATE_H
