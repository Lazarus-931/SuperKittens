#ifndef SK_RMSNORM_H
#define SK_RMSNORM_H
#include <cstdint>
extern "C" {
int sk_rmsnorm(void* x, void* gamma, void* y, uint32_t rows, uint32_t d, float eps);
}
#endif
