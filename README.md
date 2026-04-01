## SuperKitten


<img alt="image of a cute cat standing fearless" height="300" src="meow.png" title="cute cat" width="200"/>

**TK inspired kernels in Apple's Metal Shading Language!**


## Why?

Writing deep learning metal kernels should be easy; this library aims to do such that, without sacrificing performance.

It is:
1. ** Simple **

SuperKittens is straightforward to write and works seamlessly out the box with your existing apple silicon code running 
on any of the M(1, 2, 3, 4, 5) chips.

2. ** Fast **

The aim was never sacrificing perf for easier abstractions, we didn't! In opposite, we aim to provide simpler, yet 
much faster kernels that are still performant. 




# Supported Chips
April 2026:

* We currently only support M1 and M2 and are in the process of adding support for M2+.




### Todo
- [ ] Full support for single M-series chip
  - [ ] M1
  - [ ] M2
  - [ ] M3
  - [ ] M4
  - [ ] M5
- [ ] Fp16 Support across GEMM for all chips
- 