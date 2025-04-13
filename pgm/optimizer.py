from quadratic import quadratic
import numpy as np

def hill_climbing(start_x, a, b, c, delta=0.01, max_iter=1000):
    x = start_x  
    history = [(x, abs(quadratic(x, a, b, c)))]
    for _ in range(max_iter):

        current_val = abs(quadratic(x, a, b, c))
        left = x - delta
        right = x + delta 
        left_val = abs(quadratic(left, a, b, c))
        right_val = abs(quadratic(right, a, b, c))

        if left_val < current_val:
            x = left
        elif right_val < current_val:
            x = right
        else:
            break

        history.append((x, abs(quadratic(x, a, b, c))))
    return x, history 

def find_both_roots(a, b, c, delta=0.01, max_iter=1000):

    vertex_x = -b / (2 * a)
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return (None, []), (None, [])   
    root_separation = np.sqrt(b**2 - 4*a*c) / (2*a) if b**2 - 4*a*c >= 0 else 1 
    start_x1 = vertex_x - max(root_separation, 1) 
    start_x2 = vertex_x + max(root_separation, 1)
    root1, history1 = hill_climbing(start_x1, a, b, c, delta, max_iter)
    root2, history2 = hill_climbing(start_x2, a, b, c, delta, max_iter)
    print(history1[-1], history2[-1])
    return (root1, history1), (root2, history2)
