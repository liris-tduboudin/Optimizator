import jax.numpy as jnp
import jax.random as random
    
def domain_sampler(nb_points, dims, seed):
    magnitude = 1.0
    key = random.PRNGKey(seed)
    return magnitude * 2.0 * (random.uniform(key, (nb_points, dims)) - 0.5)

if __name__ == '__main__':

    nb_points = 100
    dims = 41

    for k in range (10):

        x = domain_sampler(nb_points, dims, k)
        print(x[:, 0])
