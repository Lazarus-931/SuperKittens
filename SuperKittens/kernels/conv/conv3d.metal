//
//  conv3d.metal — 3D convolution (NDHWC, fp16)
//  Scalar direct convolution, one thread per output voxel+channel.
//

#include <metal_stdlib>
using namespace metal;

[[host_name("conv3d")]]
[[kernel]]
void conv3d(
    device const half* x,
    device const half* weight,
    device const half* bias,
    device half* y,
    constant uint& N, constant uint& D, constant uint& H, constant uint& W, constant uint& C_in,
    constant uint& K_D, constant uint& K_H, constant uint& K_W, constant uint& C_out,
    constant uint& D_out, constant uint& H_out, constant uint& W_out,
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_grid]])
{
    const uint flat = tid.x;
    const uint total_pixels = D_out * H_out * W_out;
    const uint pixel_idx = flat % total_pixels;
    const uint c_out = flat / total_pixels;
    const uint d_out = pixel_idx / (H_out * W_out);
    const uint hw_idx = pixel_idx % (H_out * W_out);
    const uint h_out = hw_idx / W_out;
    const uint w_out = hw_idx % W_out;
    const uint n = gid.z;
    if (n >= N || c_out >= C_out) return;

    float acc = float(bias[c_out]);
    for (uint kd = 0; kd < K_D; kd++) {
        for (uint kh = 0; kh < K_H; kh++) {
            for (uint kw = 0; kw < K_W; kw++) {
                for (uint ci = 0; ci < C_in; ci++) {
                    uint d_in = d_out + kd, h_in = h_out + kh, w_in = w_out + kw;
                    if (d_in < D && h_in < H && w_in < W) {
                        acc += float(x[(((n * D + d_in) * H + h_in) * W + w_in) * C_in + ci]) *
                               float(weight[(((kd * K_H + kh) * K_W + kw) * C_in + ci) * C_out + c_out]);
                    }
                }
            }
        }
    }
    y[(((n * D_out + d_out) * H_out + h_out) * W_out + w_out) * C_out + c_out] = half(acc);
}
