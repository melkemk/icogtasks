import matplotlib.pyplot as plt
import numpy as np
from quadratic import quadratic

def plot_results(a, b, c, root1_data, root2_data, true_roots, filename='quadratic_hill_climbing_plot.png'):
    root1, history1 = root1_data
    root2, history2 = root2_data 

    x_vals = np.linspace(-5, 5, 400)
    y_vals = quadratic(x_vals, a, b, c)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) 

    ax1.plot(x_vals, y_vals, label=f'f(x) = {a}x² + {b}x + {c}')
    ax1.axhline(0, color='black', linestyle='--', alpha=0.3)

    if true_roots:
        ax1.axvline(true_roots[0], color='green', linestyle='--', label='True roots', alpha=0.5)
        ax1.axvline(true_roots[1], color='green', linestyle='--', alpha=0.5)

    history1_x = [h[0] for h in history1] 
    history1_y = [quadratic(h[0], a, b, c) for h in history1]
    ax1.plot(history1_x, history1_y, 'ro-', label='Path to root 1')

    history2_x = [h[0] for h in history2]
    history2_y = [quadratic(h[0], a, b, c) for h in history2] 
    ax1.plot(history2_x, history2_y, 'bo-', label='Path to root 2')

    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Hill-Climbing Paths to Roots')
    ax1.legend()

    steps1 = range(len(history1))
    errors1 = [h[1] for h in history1]
    steps2 = range(len(history2))
    errors2 = [h[1] for h in history2]
    ax2.plot(steps1, errors1, 'ro-', label='|f(x)| for root 1')
    ax2.plot(steps2, errors2, 'bo-', label='|f(x)| for root 2')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('|f(x)|')
    ax2.set_title('Convergence of |f(x)|')
    ax2.set_yscale('log')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close() 
