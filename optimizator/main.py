# import jax
# jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pickle 

import function.config as func_config
from function.helper_functions import construction_Zred_Fred_dZreddw_dFreddw
from function.target_function import target_function_maker
from function.domain_sampler import domain_sampler

from optimizer.optimize import optimize
import optimizer.config as opt_config

from logger.visualize import visualize

import time
from datetime import timedelta

if __name__ == '__main__':

    if opt_config.checkpoint_path is not None:
        checkpoint = pickle.load(opt_config.checkpoint_path)
        omega_start = checkpoint['omega_start']
        omega_end = checkpoint['omega_end']
        omega_steps = checkpoint['omega_steps']
        solutions_storage = checkpoint['solutions_storage']
    else:
        omega_start = func_config.omegas_range[0]
        omega_end = func_config.omegas_range[1]
        omega_steps = func_config.omegas_steps
        solutions_storage = []
    
    global_start_time = time.time()

    for omega_idx, omega in enumerate(np.linspace(omega_start, omega_end, omega_steps)):

        omega_start_time = time.time()

        Z, Fh = construction_Zred_Fred_dZreddw_dFreddw(omega,func_config.M,func_config.C,func_config.K,func_config.F,func_config.Nh,func_config.beta,func_config.gamma,func_config.ddl_ln,func_config.ddl_nl,func_config.derivatives,func_config.penalite,func_config.nu)

        target_function_hparams = {'Z':Z, 'Fh':Fh, 'nb_dims':2*func_config.Nh+1}
        target_function = target_function_maker(target_function_hparams)

        optimization_hparams = {'nb_points':opt_config.nb_points,
                                'iterations':opt_config.iterations,
                                'kept_threshold':opt_config.kept_threshold,
                                'merge_threshold':opt_config.merge_threshold,
                                'max_history':opt_config.max_history,
                                'tolerance_change':opt_config.tolerance_change,
                                'seed':omega_idx}

        solutions, loss = optimize(target_function, target_function_hparams, domain_sampler, optimization_hparams)
        if solutions is not None:
            solutions_storage.append([omega, solutions])
            checkpoint = {'omega_start':omega+(func_config.omegas_range[1]-func_config.omegas_range[0])/func_config.omegas_steps,
                          'omega_end':func_config.omegas_range[1],
                          'omega_steps':func_config.omegas_steps-omega_idx-1,
                          'solutions_storage':solutions_storage}
            with open('./checkpoint/checkpoint.pt', 'wb') as file:
                pickle.dump(checkpoint, file)
            print("For omega =", omega, ",", solutions.shape[0], "solutions were found.")
            print("With losses :", loss)
        else:
            print("For omega =", omega, ",", "no solutions were found.")
            print("Best loss reached :", loss)

        omega_end_time = time.time()
        
        time_since_global_start = omega_end_time - global_start_time
        print("Time elapsed since beginning :", str(timedelta(seconds=time_since_global_start)))
        current_omega_duration = omega_end_time-omega_start_time
        print("Current omega optimization duration :", str(timedelta(seconds=current_omega_duration)))
        if omega_idx == 0:
            avg_omega_duration = current_omega_duration
        else:
            avg_omega_duration = (omega_idx * avg_omega_duration + current_omega_duration)/(omega_idx+1)
        print("Expected remaining time :", str(timedelta(seconds=avg_omega_duration*(omega_steps-omega_idx-1))))
        print("------------")


    amps_lin = []
    for omega, solutions in solutions_storage:
        solutions_amps_lin = []
        for solution_idx in range(solutions.shape[0]):
            solution = solutions[solution_idx]
            x_t = func_config.IDFT_1ddl@solution
            solutions_amps_lin.append(jnp.abs(x_t).max())
        amps_lin.append([omega, solutions_amps_lin])

    visualize(amps_lin)

