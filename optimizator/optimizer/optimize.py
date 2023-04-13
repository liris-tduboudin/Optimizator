import torch
import torch.optim as optim

def optimize(target_function, 
             target_function_hparams,
             domain_sampler,
             optimization_hparams,
             device
             ):

    # definitions
    points = domain_sampler(optimization_hparams['nb_points'], target_function_hparams['nb_dims']).to(torch.device(device))
    
    # optimizer = optim.SGD(points.parameters(), optimization_hparams['learning_rate'], momentum=0.0)
    optimizer = optim.Adam(points.parameters(), optimization_hparams['learning_rate'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     factor=optimization_hparams['scheduler_factor'], 
                                                     patience=optimization_hparams['scheduler_patience'], 
                                                     threshold=1e-8,
                                                     min_lr=optimization_hparams['scheduler_min_lr'])

    # optimization loop
    for iteration in range(optimization_hparams['iterations']):

        optimizer.zero_grad()

        function_outputs = target_function(target_function_hparams, points(-1), device)

        loss = torch.linalg.vector_norm(function_outputs, dim=1).mean()

        loss.backward()

        if iteration % 1000 == 0:
            print(iteration, loss.item())

        optimizer.step()

        scheduler.step(loss)

    # removal of improper solutions (e.g. bad local minima or saddle points)
    non_reduced_loss = torch.linalg.vector_norm(function_outputs, dim=1)
    valid_solutions_indices = (non_reduced_loss < optimization_hparams['kept_threshold']).float().nonzero()
    valid_solutions = points(-1)[valid_solutions_indices.squeeze()]

    # if no solutions are found at all
    if valid_solutions.numel() == 0:
        return None, torch.linalg.vector_norm(target_function(target_function_hparams, points(-1), device)).min().item()

    # merge of identical (up to an distance threshold) solutions
    # done with a loop for memory consumption reasons
    distinct_solutions = valid_solutions[0].unsqueeze(0)
    for valid_solution_idx in range(valid_solutions.size(1)):
        distance_to_all_found_distinct_solutions = torch.cdist(distinct_solutions, valid_solutions[valid_solution_idx].view(1,1,target_function_hparams['nb_dims']))
        # if a new distinct solution is found : add it to the list
        if (distance_to_all_found_distinct_solutions > optimization_hparams['merge_threshold']).all():
            distinct_solutions = torch.cat(distinct_solutions, valid_solutions[valid_solution_idx].unsqueeze(0), dim=0)
        # else : add the solution to its already found corresponding one for better approximation using an exponential average
        else:
            corresponding_distinct_solution_idx = (distance_to_all_found_distinct_solutions < optimization_hparams['merge_threshold']).float().nonzero()
            distinct_solutions[corresponding_distinct_solution_idx] = optimization_hparams['exponential_mean'] * distinct_solutions[corresponding_distinct_solution_idx] + (1-optimization_hparams['exponential_mean']) * valid_solutions[valid_solution_idx]

    return distinct_solutions.cpu(), torch.linalg.vector_norm(target_function(target_function_hparams, distinct_solutions.to(device), device)).flatten().tolist()

    