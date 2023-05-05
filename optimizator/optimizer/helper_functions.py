import jax
import jax.numpy as jnp

def derivatives(target_function):
    # target_function_hessian = jax.jit(jax.jacfwd(jax.jacrev(target_function_with_current_parameters)))
    # target_function_gradient = jax.jit(jax.grad(target_function_with_current_parameters))
    target_function_hessian = jax.jacfwd(jax.jacrev(target_function))
    target_function_gradient = jax.grad(target_function)
    return target_function_gradient, target_function_hessian

def exact_newton_step(x, hessian, gradient):
    return x - 1e-2 * jnp.linalg.solve(hessian, gradient)

@jax.jit
def jax_pairwise_distance(x,y):
    x2 = jnp.repeat(jnp.expand_dims(jnp.sum(x**2, axis=1), axis=1), y.shape[0], axis=1)
    y2 = jnp.repeat(jnp.expand_dims(jnp.sum(y**2, axis=1), axis=1), x.shape[0], axis=1).transpose()
    xy = - 2 * jnp.matmul(x, y.transpose())
    return x2+y2+xy

def make_BFGS_step(gradient_function):
    def BFGS_step(x, approx_inv_hessian):
        current_gradient = gradient_function(x)
        current_step = - approx_inv_hessian@current_gradient
        next_x = x + current_step
        next_gradient = gradient_function(next_x)
        y = jnp.expand_dims(next_gradient - current_gradient, axis=1)
        s = jnp.expand_dims(current_step, axis=1)
        Id = jnp.identity(approx_inv_hessian.shape[0], dtype=jnp.float64)
        next_approx_inv_hessian = (Id - s@y.transpose()/(y.transpose()@s+1e-9))@approx_inv_hessian@(Id - y@s.transpose()/(y.transpose()@s+1e-9)) + s@s.transpose()/(y.transpose()@s+1e-9)
        return next_x, next_approx_inv_hessian
    return BFGS_step