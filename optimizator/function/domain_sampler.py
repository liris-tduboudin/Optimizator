import jax.numpy as jnp
import jax.random as random
    
def domain_sampler(nb_points, dims):

    magnitude = 1.0
    key = random.PRNGKey(nb_points*dims-1)
    return magnitude * 2.0 * (random.uniform(key, (nb_points, dims)) - 0.5)