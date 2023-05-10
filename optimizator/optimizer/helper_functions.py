import jax
import jax.numpy as jnp

def exact_newton_step(x, hessian, gradient):
    return x - jnp.linalg.solve(hessian, gradient)

@jax.jit
def jax_pairwise_distance(x,y):
    x2 = jnp.repeat(jnp.expand_dims(jnp.sum(x**2, axis=1), axis=1), y.shape[0], axis=1)
    y2 = jnp.repeat(jnp.expand_dims(jnp.sum(y**2, axis=1), axis=1), x.shape[0], axis=1).transpose()
    xy = - 2 * jnp.matmul(x, y.transpose())
    return x2+y2+xy

def lbfgs_step(x, g, s, y):
    q = g
    alpha = jnp.zeros(s.shape[0])
    mu = 1/((s*y).sum(axis=1)+1e-15)
    for i in range(0, s.shape[0], 1):
        alpha_i = mu[i] * jnp.dot(s[i], q)
        alpha = alpha.at[i].set(alpha_i)
        q = q - alpha_i * y[i]
    B = jnp.identity(x.shape[0]) * jnp.dot(s[0], y[0])/(jnp.dot(y[0], y[0])+1e-15)
    z = B.dot(q)
    for i in range(s.shape[0]-1, -1, -1):
        beta = mu[i] * jnp.dot(y[i], z)
        z = z + s[i] * (alpha[i] - beta)
    next_x = x - z
    return next_x, -z

if __name__ == '__main__':

    # unbatched version

    key = jax.random.PRNGKey(-1)
    new_key, key = jax.random.split(key)
    t = jax.random.uniform(key, (7,))
    x = 100 * jax.random.uniform(new_key, (7,))

    def l2(u):
        return ((u-t)**2).sum()

    l2_grad = jax.grad(l2)

    init_g = l2_grad(x)
    x = x - init_g
    next_g = l2_grad(x)
    s = jnp.expand_dims(-init_g, axis=0)
    y = jnp.expand_dims(next_g-init_g, axis=0)

    for k in range(20):

        g = next_g
        x, update = lbfgs_step(x, g, s, y)
        next_g = l2_grad(x)
        s = jnp.concatenate((jnp.expand_dims(update, axis=0), s), axis=0)
        y = jnp.concatenate((jnp.expand_dims(next_g-g, axis=0), y), axis=0)
        if s.shape[0] >= 3:
            s = s[:-1, :]
            y = y[:-1, :]
        print(l2(x))
