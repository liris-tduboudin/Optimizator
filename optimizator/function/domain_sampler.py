import jax.numpy as jnp
import jax.random as random
    
def domain_sampler(nb_points, dims, key):
    magnitude = 1.0
    key = random.PRNGKey(key)
    return magnitude * 2.0 * (random.uniform(key, (nb_points, dims), dtype=jnp.float64) - 0.5)