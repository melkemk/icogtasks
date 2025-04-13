import numpy as np
from optimizer import find_both_roots
from utils import compute_true_roots
from plotter import plot_results
from quadratic import quadratic

def main():
    np.random.seed(42)
    a, b, c = 1,5,6 # Coefficients for the quadratic equation ax^2 + bx + c = 0
    
    (root1, history1), (root2, history2) = find_both_roots(a, b, c, delta=0.01, max_iter=1000)
    true_roots_result = compute_true_roots(a, b, c)
    
    plot_results(a, b, c, (root1, history1), (root2, history2), true_roots_result)
    
    if root1 is not None: 
        print(f"Found root 1: x ≈ {root1:.4f}, f(x) ≈ {quadratic(root1, a, b, c):.4e}")
    else:
        print("Root 1 could not be found.")
    
    if root2 is not None:
        print(f"Found root 2: x ≈ {root2:.4f}, f(x) ≈ {quadratic(root2, a, b, c):.4e}")
    else:
        print("Root 2 could not be found.")
    if true_roots_result:
        print(f"True roots: x = {true_roots_result[0]:.4f}, {true_roots_result[1]:.4f}")
    else:
        print("No real roots exist.")

if __name__ == "__main__":
    main()
