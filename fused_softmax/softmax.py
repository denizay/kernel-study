import torch
import triton
import triton.language as tl

import torch.nn.functional as F

DEVICE = torch.device("cuda", index=0)


@triton.jit
def softmax_kernel(
    in_ptr,
    out_ptr,
    num_rows,
    num_cols,
    BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0)
    block_start = program_id * num_cols
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = tl.arange(0, BLOCK_SIZE) < num_cols

    x = tl.load(in_ptr + offsets, mask=mask, other=-float('inf'))
    max_vals = tl.max(x)
    x = x - max_vals
    x = tl.exp(x)
    sums = tl.sum(x)
    out = x / sums
    tl.store(out_ptr + offsets, out, mask=mask)


def softmax(
    x: torch.Tensor,
):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    softmax_kernel[(n_rows,)](x, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output


def naive_softmax(x: torch.Tensor):
    max_vals = torch.max(x, dim=1)
    x = x - max_vals.values[:, None]
    x = torch.exp(x)
    sums = torch.sum(x, dim=1)
    x = x / sums[:, None]
    return x


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'], 
        x_vals=[128 * i for i in range(120, 140)],
        line_arg='provider', 
        line_vals=['triton', 'torch', 'naive_softmax'], 
        line_names=["Triton", "Torch", "Naive Softmax"],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel="GB/s",
        plot_name="softmax-performance", 
        args={'M': 4096},
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    # gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return ms


def main():
    benchmark.run(show_plots=False, print_data=True)
    print()
    for shape in [(3,4), (32, 129), (4096,128), (5, 142), (4096, 128*10)]:
        x = torch.randn(shape, device=DEVICE)
        
        sm_torch = F.softmax(x, dim=1)
        sm_triton = softmax(x)

        max_diff = torch.max(torch.abs(sm_torch - sm_triton)).item()
        all_close = torch.allclose(sm_triton, sm_torch, atol=1e-5, rtol=1e-5)
        print(f"{all_close=} {max_diff=}")
    


if __name__ == "__main__":
    main()