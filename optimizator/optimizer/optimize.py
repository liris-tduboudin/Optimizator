import jax
import jax.numpy as jnp
from optimizer.helper_functions import derivatives, exact_newton_step, jax_pairwise_distance, make_BFGS_step

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
    inputs = domain_sampler(optimization_hparams['nb_points'], target_function_hparams['nb_dims'], optimization_hparams['key'])

    vectorized_target_function = jax.jit(jax.vmap(target_function))
    loss_function_gradient, loss_function_hessian = derivatives(loss_function)
    vectorized_loss_function_gradient = jax.jit(jax.vmap(loss_function_gradient))
    vectorized_loss_function_hessian = jax.jit(jax.vmap(loss_function_hessian))

    ### exact newton ###
    vectorized_exact_newton_step = jax.jit(jax.vmap(exact_newton_step))

    ### bfgs ###
    # bfgs_step = make_BFGS_step(loss_function_gradient)
    # vectorized_bfgs_step = jax.jit(jax.vmap(bfgs_step))
    # approx_hessians = jnp.repeat(jnp.expand_dims(jnp.identity(target_function_hparams['nb_dims']), axis=0), optimization_hparams['nb_points'], axis=0)
    # vectorized_inv = jax.jit(jax.vmap(jnp.linalg.inv))
    # approx_inv_hessians = vectorized_inv(vectorized_loss_function_hessian(inputs))

    # optimization loop
    for iteration in range(optimization_hparams['iterations']):

        ### exact newton ###
        hessians = vectorized_loss_function_hessian(inputs)
        gradients = vectorized_loss_function_gradient(inputs)
        inputs = vectorized_exact_newton_step(inputs, hessians, gradients)

        ### bfgs ###
        # inputs, approx_inv_hessians = vectorized_bfgs_step(inputs, approx_inv_hessians)

        if iteration % 1000 == 0:
            losses = (vectorized_target_function(inputs)**2).sum(axis=1)
            # print(losses)
            print(losses.mean())

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

    