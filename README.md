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

### Jun 9 2026 | Transpose
Okay, I can transpose really fast, that's cool but my naive gpu implementation is faster than my "improved" gpu implementation with memory coalescing and no memory bank conflicts. Gemini says "Your v2 kernel is perfectly written." but l1 cache is too big and smart on my gpu so naive just runs faster. I'm not sure how much I believe.
-   Do this in Triton as well
-   See if there's a method/tool to examine possible bank conflicts and utilization of the memory transactions.
-   Same as yesterday, improve CPU version.

### Jun 10 2026 | Softmax
Short entry cuz 2AM. Used Triton. Up to 16384 elements per row mine is faster than F.softmax, after that registers spill on my 2080ti. Implementation on tritons website has more tricks will try them tomorrow.

### Jun 11 2026 | Softmax, persistent kernels and occupancy
Version of softmax calculates register and shared memory usage to calculate occupancy to find "optimal" number of programs and keeps them persistend and uses pipeline staging. I think my 2080ti doesn't support staging though. Version 2 is not really faster. Compared to triton official tutorial one and looks same.
-   Compare implementations on other GPUs
-   Online softmax to handle big rows w/o register spilling (Good warm up for flash attention)

### Jun 15 2026 | Matmul
Back to cuda again. It was fun. A naive cuda took 0.016265 secs on my 2080ti for two 2048 by 2048 matrices and thru different versions got it down to 0.003835secs. Used shared memory tiling and register tiling, 2D. 
-   Examine why SIZE_RIGHT_BLOCK = 2 & SIZE_LEFT_BLOCK = 4 performs best. Profile.
-   Should use better metrics then "n secs on my 2080ti".
-   Matmul on Triton.
-   Compare with the implementations https://siboehm.com/articles/22/CUDA-MMM here.