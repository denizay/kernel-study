### Jun 7 2026 | The plan
Implement kernels, in CUDA and Triton, starting simple and moving to more complex ones. Have different versions of the same kernel trying out different techniques. Compare performance.

### Jun 8 2026 | Sum reduction
Studying parallel reductions, specifically sum reduction.
Naive serial method on my cpu takes 0.227665 seconds to process 100000000 elements.
Can achieve 0.027055 on my 2080ti right now.
Don't know how to do inter thread-block communication so it's using a single SM right now. Most of the GPU is idle ㅠㅠ.
-   Figure out how to use multiple thread blocks to unlock the power of all SMs.
-   Compare with Triton, simple `tl.sum()` would crush this.
-   We can try to implement this in Triton without `tl.sum()`
-   If bored can try optimizing CPU version as well for funsies. Actually could be informative.