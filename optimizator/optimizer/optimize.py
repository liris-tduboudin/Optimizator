import jax
import jax.numpy as jnp
from optimizer.helper_functions import exact_newton_step, lbfgs_step, jax_pairwise_distance

def optimize(target_function, 
             target_function_hparams,
             domain_sampler,
             optimization_hparams
             ):

    def loss_function(x):
        y = target_function(x)
        loss = (y**2).sum()
        return loss

    # definitions
    inputs = domain_sampler(optimization_hparams['nb_points'], target_function_hparams['nb_dims'], optimization_hparams['seed'])

    # used for all methods
    vectorized_target_function = jax.jit(jax.vmap(target_function))
    loss_function_gradient = jax.grad(loss_function)
    vectorized_loss_function_gradient = jax.jit(jax.vmap(loss_function_gradient))

    ### exact newton ###
    # loss_function_hessian = jax.jacfwd(jax.jacrev(target_function))
    # vectorized_loss_function_hessian = jax.jit(jax.vmap(loss_function_hessian))
    # vectorized_exact_newton_step = jax.jit(jax.vmap(exact_newton_step))

    ### lbfgs ###
    vectorized_lbfgs_step = jax.jit(jax.vmap(lbfgs_step))
    init_g = vectorized_loss_function_gradient(inputs)
    inputs = inputs-init_g
    next_g = vectorized_loss_function_gradient(inputs)
    s = jnp.expand_dims(-init_g, axis=1)
    y = jnp.expand_dims(next_g-init_g, axis=1)

    # optimization loop
    for iteration in range(optimization_hparams['iterations']):

        ### exact newton ###
        # hessians = vectorized_loss_function_hessian(inputs)
        # gradients = vectorized_loss_function_gradient(inputs)
        # inputs = vectorized_exact_newton_step(inputs, hessians, gradients)

        ### lbfgs ###
        g = next_g
        inputs, updates = vectorized_lbfgs_step(inputs, g, s, y)
        next_g = vectorized_loss_function_gradient(inputs)
        s = jnp.concatenate((jnp.expand_dims(updates, axis=1), s), axis=1)
        y = jnp.concatenate((jnp.expand_dims(next_g-g, axis=1), y), axis=1)
        if s.shape[1] >= optimization_hparams['max_history']:
            s = s[:, :-1, :]
            y = y[:, :-1, :]

        if iteration % 100 == 0:
            losses = (vectorized_target_function(inputs)**2).sum(axis=1)
            print("avg loss :", losses.mean())

    # removal of improper solutions (e.g. bad local minima or saddle points)
    non_reduced_loss = (vectorized_target_function(inputs)**2).sum(axis=1)
    valid_solutions_indices = jnp.nonzero((non_reduced_loss < optimization_hparams['kept_threshold']).astype(dtype=jnp.int8))
    valid_solutions = jnp.reshape(inputs[valid_solutions_indices[0]], (valid_solutions_indices[0].shape[0], target_function_hparams['nb_dims']))
    # if no solutions are found at all
    if valid_solutions.size == 0:
        return None, non_reduced_loss.min()

    # merge of identical (up to an distance threshold) solutions
    # done with a loop for memory consumption reasons
    distinct_solutions = jnp.expand_dims(valid_solutions[0], axis=0)
    for valid_solution_idx in range(1,valid_solutions.shape[0]):
        distance_to_all_found_distinct_solutions = jax_pairwise_distance(distinct_solutions, jnp.expand_dims(valid_solutions[valid_solution_idx], axis=0))
        # if a new distinct solution is found : add it to the list
        if (distance_to_all_found_distinct_solutions > optimization_hparams['merge_threshold']).all():
            distinct_solutions = jnp.concatenate((distinct_solutions, jnp.expand_dims(valid_solutions[valid_solution_idx], axis=0)), axis=0)

    return distinct_solutions, (vectorized_target_function(distinct_solutions)**2).sum(axis=1)
    