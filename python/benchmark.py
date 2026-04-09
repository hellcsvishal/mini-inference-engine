import engine
import time
import random

N = 256
M = 256

W_2d = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(M)]
x = [random.uniform(-1, 1) for _ in range(N)]

W_flat = [val for row in W_2d for val in row]

def run(name, fn):
    start = time.time()
    for _ in range(100):
        fn()
    end = time.time()
    print(f"{name:15}: {end - start:.4f} sec")


print("\n=== Advanced Benchmark ===")

run("FP32", lambda: engine.matvec_flat(W_flat, x, M, N))
run("Unrolled", lambda: engine.matvec_unrolled(W_flat, x, M, N))
run("AVX", lambda: engine.matvec_avx(W_flat, x, M, N))
