import numpy as np
import torch

import function.config as func_config
from function.helper_functions import construction_Zred_Fred_dZreddw_dFreddw
from function.target_function import target_function
from function.domain_sampler import DomainSampler

from optimizer.optimize import optimize
import optimizer.config as opt_config

from logger.visualize import visualize

import time
from datetime import timedelta

if __name__ == '__main__':

    device = torch.device(opt_config.device)

    if opt_config.checkpoint_path is not None:
        checkpoint = torch.load(opt_config.checkpoint_path)
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
        Z = torch.tensor(Z, dtype=torch.float32, device=device)
        Fh = torch.tensor(Fh, dtype=torch.float32, device=device)

        target_function_hparams = {'Z':Z, 'Fh':Fh, 'nb_dims':2*func_config.Nh+1}

        optimization_hparams = {'learning_rate':opt_config.learning_rate,
                                'nb_points':opt_config.nb_points,
                                'iterations':opt_config.iterations,
                                'kept_threshold':opt_config.kept_threshold,
                                'merge_threshold':opt_config.merge_threshold,
                                'exponential_mean':opt_config.exponential_mean,
                                'scheduler_factor':opt_config.scheduler_factor,
                                'scheduler_patience':opt_config.scheduler_patience,
                                'scheduler_min_lr':opt_config.scheduler_min_lr,
                                'finetuning_iterations':opt_config.finetuning_iterations}

        solutions, loss = optimize(target_function, target_function_hparams, DomainSampler, optimization_hparams, device)
        if solutions is not None:
            solutions_storage.append([omega, solutions])
            checkpoint = {'omega_start':omega+(func_config.omegas_range[1]-func_config.omegas_range[0])/func_config.omegas_steps,
                          'omega_end':func_config.omegas_range[1],
                          'omega_steps':func_config.omegas_steps-omega_idx-1,
                          'solutions_storage':solutions_storage}
            torch.save(checkpoint, './checkpoint/checkpoint.pt')
            print("For omega =", omega, ",", solutions.size(0), "solutions were found.")
            print("With losses :", loss)
        else:
            print("For omega =", omega, ",", "no solutions were found.")
            print("Best loss reached :", loss)

        omega_end_time = time.time()
        omega_duration = omega_end_time-omega_start_time
        print("Time elapsed :", str(timedelta(seconds=omega_duration)))
        print("Expected remaining time :", str(timedelta(seconds=omega_duration*(omega_steps-omega_idx-1))))

    amps_lin = []
    for omega, solutions in solutions_storage:
        solutions_amps_lin = []
        for solution_idx in range(solutions.size(0)):
            solution = solutions[solution_idx]
            x_t = torch.mv(func_config.IDFT_1ddl_tensor, solution)
            solutions_amps_lin.append(torch.max(x_t.abs()).item())
        amps_lin.append([omega, solutions_amps_lin])

    visualize(amps_lin)

