import time
import torch


def f(x, y):
    a = x * 2
    b = y + 3
    c = a + b
    d = torch.relu(c)
    return d


def run_once(fn, x, y, warmup=10, iters=50):
    for _ in range(warmup):
        _ = fn(x, y)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn(x, y)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This demo is intended for CUDA. No GPU found.")

    device = "cuda"
    x = torch.randn(2048, 2048, device=device)
    y = torch.randn(2048, 2048, device=device)

    f_compiled = torch.compile(f)

    eager_time = run_once(f, x, y)
    compiled_time = run_once(f_compiled, x, y)

    print(f"Eager avg step time:    {eager_time * 1e3:.3f} ms")
    print(f"Compiled avg step time: {compiled_time * 1e3:.3f} ms")
    print("Tip: run with TORCH_LOGS='+dynamo,inductor' to inspect compile behavior.")


if __name__ == "__main__":
    main()
