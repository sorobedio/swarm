
import os

# from PyQt5.QtGui.QRawFont import weight

from helpers.helpers import *
import torch
import lm_eval
# from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import evaluator, utils
from lm_eval.evaluator import request_caching_arg_to_dict
from lm_eval.loggers import EvaluationTracker, WandbLogger
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable, make_table, simple_parse_args_string
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
import random
import numpy as np
import argparse
import argparse, os, sys, datetime, glob

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_grad_enabled(False)

# Utility function

# Utility function
def utility_function(wd, layer):
    # Example utility function (sum of weights for demonstration)
    num_samples = wd.shape[0]
    utility_value = []
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map=device,
                                                 )
    # layers = list(wd)
    # num_samples = num_samples
    # layer = "Llama-3.2-1B-Instruct"
    for i in range(num_samples):
        wr = {}
        # for l in layer:
        wr = wd[i].reshape(-1)

        std = model.state_dict()

        std = set_layer_state_dict(std, wr, layer='norm')
        model.load_state_dict(std)

        # std = set_layers_state_dict(std, wr)
        # model.load_state_dict(std)

        print('---------evaluating model-----------------------------')

        lm_eval_model = HFLM(model, device=device, batch_size='auto')

        task_manager = lm_eval.tasks.TaskManager()

        with torch.no_grad():
            results = lm_eval.simple_evaluate(
                model=lm_eval_model,
                tasks=["arc_challenge"],
                num_fewshot=0,
                # apply_chat_template=True,
                # fewshot_as_multiturn=True,
                limit=200,
                batch_size=64,
                task_manager=task_manager,
            )

        #
        # results = lm_eval.simple_evaluate(  # call simple_evaluate
        #     model=lm_eval_model,
        #     tasks=["winogrande"],
        #     num_fewshot=5,
        #     apply_chat_template=True,
        #     fewshot_as_multiturn=True,
        #     limit=200,
        #     # output_base_path="results_Out",
        #     task_manager=task_manager,
        # )
        mtable = make_table(results)
        print(mtable)
        # Get the current date and format it
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create the header with asterisks and the date
        header = f"\n{'*' * 40}\n{current_date}\n{'*' * 40}\n"
        file_path = 'logfiles/swamarc_llamaparticles_results_v2.txt'
        # Append the header and the Markdown table to the file
        with open(file_path, 'a') as file:
            file.write(header)
            file.write(mtable)

            file.write('\n')  # Add a new line at the end
            file.write(f'-------iteration--{i}-------\n')

        print('****************************************')
        if "groups" in results:
            print(make_table(results, "groups"))
        print(results['results'])
        # acc = results['results']['mmlu']['acc,none']
        # acc = results['results']['gsm8k']['exact_match,strict-match']
        acc = results['results']['arc_challenge']['acc_norm,none']
        utility_value.append(acc)

    return torch.tensor(utility_value)

# def initialize_velocity_vectors(expanded_experts):
#     if not isinstance(expanded_experts, list):
#         expanded_experts = [[expanded_experts[i, :] for i in range(expanded_experts[0])]]
#     velocities = []
#     for i, expert in enumerate(expanded_experts):
#         # Randomly choose two experts to compute the velocity as a difference of their vectors
#         expert1, expert2 = np.random.choice(expanded_experts, 2, replace=False)
#         velocity = expert1 - expert2  # Velocity is the difference between two weight vectors, shape [D]
#         velocities.append(velocity)
#
#     return velocities

# Initialize weights and velocities
# wd, 'wdata/dit_sampled_weights_top1.pt'

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# model_id = "meta-llama/Llama-3.2-1B-Instruct"
model_id = "meta-llama/Llama-3.2-1B-Instruct"
# Parameters
# num_particles = 25
# weight_dim = 1000  # Dimensionality for each weight vector
lambda_step = 0.1  # Step length
varphi_lambda = 0.95  # Step length schedule
varphi_v = 0.9  # Inertia
varphi_p = 0.5  # Cognitive coefficient
varphi_g = 0.4  # Social coefficient
varphi_w = 0.3  # Repel coefficient
patience = 5  # Number of iterations to wait for no improvement in global best
restart_patience = 3  # Number of iterations for particle restart
max_iterations = 20  # Maximum number of iterations


# acc = results['results']['hellaswag']['acc_norm,none']
#         utility_values.append(acc)
#     torch.save(utility_values, 'utility_values_hella.pt')


# data = torch.load('wdata/dit_sampled_weights_top1.pt')
data = torch.load('wdata/mdt_sampled_weights_50_norm.pt')
# expanded_experts
layers = list(data)[0]
weights = data[layers]
print(weights.shape, layers)
# exit()
num_particles = weights.shape[0]
velocities = torch.zeros_like(weights)  # Initialize velocities as zeros

# Initialize global best, global worst, and personal bests
personal_best = weights.clone()
utility_values = torch.load('utility_values_arc_c.pt')
# utility_values = torch.load('utilities.pt')[layers]
utility_values = torch.tensor(utility_values).reshape((-1))
print(f'-----------{utility_values.shape}------------')
global_best = weights[torch.argmax(utility_values)].clone()
global_worst = weights[torch.argmin(utility_values)].clone()


random_indices = torch.arange(num_particles)

# Ensure each expert has a randomly selected index different from its own index
for i in range(num_particles):
    while True:
        # Pick a random index in the range of num_particles
        random_index = torch.randint(0, num_particles, (1,)).item()

        # Ensure the random index is different from the current expert's index
        if random_index != i:
            random_indices[i] = random_index
            break

# Compute initial velocities as the difference between the current expert's weight and a randomly selected expert's weight
velocities = weights[random_indices] - weights



# Search process (iterations)
for k in range(max_iterations):
    # If the global best doesn't change for a certain number of iterations, stop
    if k > patience and torch.equal(global_best, weights[torch.argmax(utility_function(weights, layers))]):
        print(f"Stopping at iteration {k} due to no improvement in global best.")
        break

    # Iterate through each particle and update positions and velocities
    for i in range(num_particles):
        # Sample random factors for inertia, cognitive, social, and repel terms
        r_v = torch.rand(1).item()
        r_p = torch.rand(1).item()
        r_g = torch.rand(1).item()
        r_w = torch.rand(1).item()

        # Update velocity according to the formula
        inertia_term = r_v * varphi_v * velocities[i]
        cognitive_term = r_p * varphi_p * (personal_best[i] - weights[i])
        social_term = r_g * varphi_g * (global_best - weights[i])
        repel_term = -r_w * varphi_w * (global_worst - weights[i])

        # Normalize the sum of the velocity components
        C = r_v * varphi_v + r_p * varphi_p + r_g * varphi_g + r_w * varphi_w
        velocities[i] = (inertia_term + cognitive_term + social_term + repel_term) / C

        # Update position: x_i = x_i + λ * v_i
        weights[i] = weights[i] + lambda_step * velocities[i]

        # Update personal best, global best, and global worst
        current_utility = utility_function(weights[i].unsqueeze(0),layers)

        # Check if the new position is better for this particle
        if current_utility > utility_function(personal_best[i].unsqueeze(0), layers):
            personal_best[i] = weights[i].clone()

        # Update global best and global worst
        if current_utility > utility_function(global_best.unsqueeze(0), layers):
            global_best = weights[i].clone()

        if current_utility < utility_function(global_worst.unsqueeze(0), layers):
            global_worst = weights[i].clone()


        # Track how many iterations each particle has gone without improvement
        stagnation_counter = torch.zeros(num_particles, dtype=torch.int)

        # Restart particles if their utility hasn't improved for `restart_patience` iterations
        if current_utility <= utility_function(personal_best[i].unsqueeze(0), layers):
            stagnation_counter[i] += 1
        else:
            stagnation_counter[i] = 0  # Reset counter if improvement is detected

        if stagnation_counter[i] >= restart_patience:
            # Apply a perturbation to weights and velocities to explore a new direction
            weights[i] = personal_best[i].clone() + torch.randn_like(
                personal_best[i]) * 0.1  # Example perturbation scale
            velocities[i] = torch.randn_like(velocities[i])  # Reinitialize velocity with randomness
            stagnation_counter[i] = 0  # Reset the counter after restart

    # Step length scheduling: λ = λ * φλ
    lambda_step = lambda_step * varphi_lambda
    torch.save(weights, f"./wdata/llama3_arc_swarm_weights_.pt")

# Output the best-found expert (global best)
# torch.save(weights, f"./wdata/llama3_arc_swarm_weights_.pt")
# print("Best-found weights (global best):", global_best)
torch.save(global_best, f"./wdata/arc_swarm_global_best_top_1.pt")
print("Utility of best-found weights:", utility_function(global_best.unsqueeze(0), layers))
