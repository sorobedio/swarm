import os
import numpy as np
import torch

def normalize_within_range(value, lower_bound, higher_bound):
    diff = value - lower_bound
    if diff > 0:
        return (value - lower_bound) / (higher_bound - lower_bound)
    else:
        return 0.0

def normalization(task, value_dict):
    if task == 'leaderboard_gpqa':
        lower_bound, higher_bound = 1/4, 1.0
        avg =0.0
        for key, value in value_dict.items():
            avg += normalize_within_range(value_dict[key], lower_bound, higher_bound)
        avg /= len(value_dict)

        return avg
    elif task == 'leaderboard_bbh':
        sub_tasks = {'leaderboard_bbh_boolean_expressions': 2,
                     'leaderboard_bbh_causal_judgement': 2,
                     'leaderboard_bbh_date_understanding': 6,
                     'leaderboard_bbh_disambiguation_qa': 3,
                     'leaderboard_bbh_dyck_languages': 2,
                     'leaderboard_bbh_formal_fallacies': 2,
                     'leaderboard_bbh_geometric_shapes': 11,
                     'leaderboard_bbh_hyperbaton': 2,
                     'leaderboard_bbh_logical_deduction_five_objects': 5,
                     'leaderboard_bbh_logical_deduction_seven_objects': 7,
                     'leaderboard_bbh_logical_deduction_three_objects': 3,
                     'leaderboard_bbh_movie_recommendation': 6,
                     'leaderboard_bbh_multistep_arithmetic_two': 2,
                     'leaderboard_bbh_navigate': 2,
                     'leaderboard_bbh_object_counting': 19,
                     'leaderboard_bbh_penguins_in_a_table': 5,
                     'leaderboard_bbh_reasoning_about_colored_objects': 18,
                     'leaderboard_bbh_ruin_names': 6,
                     'leaderboard_bbh_salient_translation_error_detection': 6,
                     'leaderboard_bbh_snarks': 2,
                     'leaderboard_bbh_sports_understanding': 2,
                     'leaderboard_bbh_temporal_sequences': 4,
                     'leaderboard_bbh_tracking_shuffled_objects_five_objects': 5,
                     'leaderboard_bbh_tracking_shuffled_objects_seven_objects': 7,
                     'leaderboard_bbh_tracking_shuffled_objects_three_objects': 3,
                     'leaderboard_bbh_web_of_lies': 2,
                     'leaderboard_bbh_word_sorting': 2}

        avg = 0.0
        for key, value in value_dict.items():
            lower_bound, higher_bound = 1/sub_tasks[key], 1.0
            avg += normalize_within_range(value_dict[key], lower_bound, higher_bound)
        avg /= len(value_dict)

        return avg

    elif task == 'leaderboard_ifeval':
        lower_bound, higher_bound = 0.0, 1.0
        avg =0.0
        for key, value in value_dict.items():
            avg += normalize_within_range(value_dict[key], lower_bound, higher_bound)
        avg /= len(value_dict)
        return avg

    elif task == 'leaderboard_math_hard':
        avg = 0.0
        lower_bound, higher_bound = 0.0, 1.0
        for key, value in value_dict.items():
            avg += normalize_within_range(value_dict[key], lower_bound, higher_bound)
        avg /= len(value_dict)
        return avg


    elif task == 'leaderboard_mmlu_pro':
        avg = 0.0
        lower_bound, higher_bound = 1/10, 1.0
        for key, value in value_dict.items():
            avg += normalize_within_range(value_dict[key], lower_bound, higher_bound)
        avg /= len(value_dict)
        return avg

    elif task == 'leaderboard_musr':
        sub_tasks = {'leaderboard_musr_murder_mysteries': 2,
                     'leaderboard_musr_object_placements': 5,
                     'leaderboard_musr_team_allocation': 3,
                     }
        avg = 0.0
        for key, value in value_dict.items():
            lower_bound, higher_bound = 1 / sub_tasks[key], 1.0
            avg += normalize_within_range(value_dict[key], lower_bound, higher_bound)
        avg /= len(value_dict)

        return avg


