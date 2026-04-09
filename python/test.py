import engine

W = [
    [0.9, -0.2, 0.0],
    [1.2, -0.8, 0.3]
]

x = [1.0, 2.0, 3.0]

# ternarize
W_t = [engine.ternarize_vector(row, 0.5) for row in W]

# run ternary matvec
result = engine.matvec_ternary(W_t, x)

print("Ternary Result:", result)
