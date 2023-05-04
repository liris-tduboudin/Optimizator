import jax 
import jax.numpy as jnp

def target_function(x):
    return jnp.dot(x, x)

vectorized_target_function = jax.vmap(target_function)

# @jax.jit
def jax_pairwise_distance(x,y):
    x2 = jnp.repeat(jnp.expand_dims(jnp.sum(x**2, axis=1), axis=1), y.shape[0], axis=1)
    y2 = jnp.repeat(jnp.expand_dims(jnp.sum(y**2, axis=1), axis=1), x.shape[0], axis=1).transpose()
    xy = - 2 * jnp.matmul(x, y.transpose())
    return x2+y2+xy

if __name__ == '__main__':

    key = jax.random.PRNGKey(0)
    key, new_key = jax.random.split(key)

    # x = jax.random.uniform(key, (1000,))
    # y = target_function(x)
    # print(y, y.shape)

    # xx = jax.random.uniform(new_key, (7, 1000))
    # yy = vectorized_target_function(xx)
    # print(yy, yy.shape)

    x = jax.random.uniform(key, (3,5))
    y = jax.random.uniform(new_key, (5,5))

    xx = jax_pairwise_distance(y, x)
    print(xx)
    print(xx.shape)