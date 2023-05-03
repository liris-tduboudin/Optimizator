import jax
import jax.numpy as jnp

def derivatives(target_function):
    # target_function_hessian = jax.jit(jax.jacfwd(jax.jacrev(target_function_with_current_parameters)))
    # target_function_gradient = jax.jit(jax.grad(target_function_with_current_parameters))
    target_function_hessian = jax.jacfwd(jax.jacrev(target_function))
    target_function_gradient = jax.grad(target_function)
    return target_function_gradient, target_function_hessian

# @jax.jit
def exact_newton_step(x, hessian, gradient):
    return x - jnp.linalg.solve(hessian, gradient)

# @jax.jit
def jax_pairwise_distance(x, y):
    return jnp.matmul(jnp.expand_dims(jnp.sum(x**2, axis=1), axis=1), jnp.expand_dims(jnp.sum(y**2, axis=1), axis=0)) - 2 * jnp.matmul(x, y.transpose())
