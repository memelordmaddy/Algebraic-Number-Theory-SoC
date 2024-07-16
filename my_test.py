import soc24mathlib

def gaussian_elimination(A, b,p):
    n = len(A[0])
    
    # Forward elimination
    for i in range(n):
        # Pivot for maximum element in column i to avoid division by zero
        max_row = max(range(i, n), key=lambda r: abs(A[i][r]))
        for k in range(n):
            A[k][i], A[k][max_row] = A[k][max_row], A[k][i]
        b[i], b[max_row] = b[max_row], b[i]
        
        # Eliminate entries below i
        for j in range(i + 1, n):
            if A[i][j] == 0:
                continue
            factor = (A[i][j] *  soc24mathlib.mod_inv(A[i][i],p))%p
            b[j] -= factor * b[i]
            b[j]%=p
            for k in range(n):
                A[k][j] -= factor * A[k][i]
                A[k][j]%=p

    # Back substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = b[i] * soc24mathlib.mod_inv(A[i][i],p)
        x[i]%=p
        for j in range(i - 1, -1, -1):
            b[j] -= A[i][j] * x[i]
            b[j]%=p
    

    return x
    


import random 
def generate_random_matrix_and_vector(n, p):
    A = [[random.randint(-(p-1), p-1) for _ in range(n)] for _ in range(n)]
    b = [random.randint(-(p-1), p-1) for _ in range(n)]
    return A, b

def test_gaussian_elimination():
    p = 1000000007  # A large prime number
    num_tests = 10
    n = 5  # Size of the matrix and vector

    for _ in range(num_tests):
        A, b = generate_random_matrix_and_vector(n, p)
        A_orig = [row[:] for row in A]  # Make a copy of A
        b_orig = b[:]  # Make a copy of b
        
        try:
            x = gaussian_elimination(A, b, p)
            
            # Verify that A*x = b (mod p)
            for i in range(n):
                Ax_i = sum(A_orig[j][i] * x[j] for j in range(n)) % p
                assert Ax_i == b_orig[i] % p, f"Test failed for matrix {A_orig} and vector {b_orig}. Got x = {x}."
            print("Test passed")
        except AssertionError as e:
            print(e)

test_gaussian_elimination()