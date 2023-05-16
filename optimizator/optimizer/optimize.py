import torch
import torch.optim as optim

def optimize(target_function, 
             target_function_hparams,
             domain_sampler,
             optimization_hparams,
             device
             ):

    # definitions
    inputs = domain_sampler(optimization_hparams['nb_points'], target_function_hparams['nb_dims']).to(torch.device(device)).requires_grad_(True)
    
    optimizer = optim.LBFGS([inputs], lr=optimization_hparams['learning_rate'], history_size=optimization_hparams['max_history'])

    def closure():
        optimizer.zero_grad()
        function_outputs = target_function(target_function_hparams, inputs)
        loss = function_outputs.pow(2).sum(dim=1).mean()
        loss.backward()
        return loss
    
    # optimization loop
    for iteration in range(optimization_hparams['iterations']):

        optimizer.step(closure)

        # if iteration % 100 == 0:
        #     loss = closure()
        #     print(iteration, loss.item())

    # removal of improper solutions (e.g. bad local minima or saddle points)
    function_outputs = target_function(target_function_hparams, inputs)
    non_reduced_loss = function_outputs.pow(2).sum(dim=1)
    valid_solutions_indices = (non_reduced_loss < optimization_hparams['kept_threshold']).float().nonzero()
    valid_solutions = inputs[valid_solutions_indices.squeeze()].view(valid_solutions_indices.numel(), target_function_hparams['nb_dims'])
    # if no solutions are found at all
    if valid_solutions.numel() == 0:
        return None, non_reduced_loss.min().item()

    # merge of identical (up to an distance threshold) solutions
    # done with a loop for memory consumption reasons
    distinct_solutions = valid_solutions[0].unsqueeze(0)
    for valid_solution_idx in range(1,valid_solutions.size(0)):
        distance_to_all_found_distinct_solutions = torch.cdist(distinct_solutions, valid_solutions[valid_solution_idx].view(1,1,target_function_hparams['nb_dims'])).pow(2)
        # if a new distinct solution is found : add it to the list
        if (distance_to_all_found_distinct_solutions > optimization_hparams['merge_threshold']).all():
            distinct_solutions = torch.cat((distinct_solutions, valid_solutions[valid_solution_idx].unsqueeze(0)), dim=0)

    return distinct_solutions.cpu(), target_function(target_function_hparams, distinct_solutions).pow(2).sum(dim=1).flatten().tolist()

    