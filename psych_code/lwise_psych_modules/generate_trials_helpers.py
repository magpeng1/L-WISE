import random
from collections import deque, defaultdict
import time
import pandas as pd


def replace_bucket_name_in_url(url, new_bucket_name):
  # Example starting URL: https://morgan-imagenet16-v1.s3.amazonaws.com/val/dog/ILSVRC2012_val_00000269.JPEG
  url_end = url.split(".s3.amazonaws.com")[1]
  return "https://" + new_bucket_name + ".s3.amazonaws.com" + url_end


def shuffle_choice_names_and_urls(choice_names_default, class_to_url_dict):
  choice_names = choice_names_default.copy()
  choice_urls = [class_to_url_dict[cl.replace(" ", "_")] for cl in choice_names]
  
  combined = list(zip(choice_names, choice_urls))
  random.shuffle(combined)
  choice_names, choice_urls = zip(*combined)
  choice_names = list(choice_names)
  choice_urls = list(choice_urls)

  return choice_names, choice_urls


def rotate_choice_names_and_urls(choice_names_default, class_to_url_dict):
  choice_names = choice_names_default.copy()

  choice_names = deque(choice_names)
  choice_names.rotate(random.randint(0, len(choice_names))-1)
  choice_names = list(choice_names)

  choice_urls = [class_to_url_dict[cl.replace(" ", "_")] for cl in choice_names]

  return choice_names, choice_urls


def get_choice_names_and_urls(block_config, trial_config, class_to_url_dict):
  default_choice_names = [cl.replace("_", " ") for cl in class_to_url_dict.keys()]

  if block_config["shuffle_choice_order"]:
    choice_names, choice_urls = shuffle_choice_names_and_urls(default_choice_names, class_to_url_dict)
  elif trial_config["choice_names_order"]:
    if block_config["rotate_choice_order"]:
      choice_names, choice_urls = rotate_choice_names_and_urls(trial_config["choice_names_order"], class_to_url_dict)
    else:
      choice_names = trial_config["choice_names_order"]
      choice_urls = [class_to_url_dict[cl.replace(" ", "_")] for cl in choice_names]
  else:
    raise ValueError("In the config.yaml, you must either specify 'shuffle_choice_order: true' or provide non-null choice_names_order (list of class names)")
  
  if "choice_names_aliases" in trial_config:
    choice_names = [trial_config["choice_names_aliases"][c] if c in trial_config["choice_names_aliases"] else c for c in choice_names]
  
  return choice_names, choice_urls


def sample_non_curriculum_trials(dataset_df, non_curric_blocks):
  non_curric_trials = pd.DataFrame()
  ds_df = dataset_df.copy(deep=True)
  for block_config in non_curric_blocks:
    for hclass in ds_df['class'].unique():
      for split in ['train', 'val', 'test']:
        n_trials = block_config[f"n_trials_per_class_{split}"]
        class_split_df = ds_df[(ds_df["split"] == split) & (ds_df["class"] == hclass)]
        sampled_df = class_split_df.sample(n=n_trials, random_state=int(time.time() % 1e6))
        len_ds_df_before = len(ds_df)
        ds_df = ds_df[~ds_df.index.isin(sampled_df.index)]
        assert len(ds_df) + len(sampled_df) == len_ds_df_before
        non_curric_trials = pd.concat([non_curric_trials, sampled_df])

  return non_curric_trials
  

def difficulty_curriculum_sample_block(df, sample_n, trial_config, session_config, block_ind, block_config):
  df_copy = df.copy(deep=True)
  df_copy = df_copy.sort_values(by="difficulty")
  if ("conditional_trial_types" in block_config and "max_difficulty_selection" in block_config["conditional_trial_types"][trial_config["condition_idx"]][next(iter(block_config["conditional_trial_types"][trial_config["condition_idx"]]))]):
    max_difficulty_selection = block_config["conditional_trial_types"][trial_config["condition_idx"]][next(iter(block_config["conditional_trial_types"][trial_config["condition_idx"]]))]["max_difficulty_selection"]
  elif "max_difficulty_selection" in block_config:
    max_difficulty_selection = block_config["max_difficulty_selection"]
  else:
    raise ValueError("max_difficulty_selection is not specified in the block configuration or trial type configuration")
  df_suit = df.iloc[:int(round(len(df)*max_difficulty_selection))]
  return df_suit.sample(n=sample_n)


def constrained_shuffle(data, preserve_property_positions=['class', 'trial_type', 'split']):
    """
    Randomly shuffles a list of dictionaries while preserving the order of specified properties.

    This function performs a constrained shuffle on a list of dictionaries (e.g., with each 
    dictionary representing 1 trial). It maintains the relative positions of specified 
    properties while randomizing the order of items within groups that share the same values for 
    these properties.

    Args:
        data (list): A list of dictionaries to be shuffled. Each dictionary should contain
                     all the keys specified in preserve_property_positions.
        preserve_property_positions (list): A list of dictionary keys whose positions should
                                            be preserved in the shuffled output. Default is
                                            ['class', 'trial_type', 'split'].

    Returns:
        list: A new list of dictionaries, shuffled according to the specified constraints.

    Behavior:
        - Items are grouped based on their values for the specified preserve_property_positions.
        - Each group is shuffled independently.
        - The original order of the preserved properties is maintained in the output list.
        - Items are shuffled with equal probability within their groups, including the
          possibility of an item remaining in its original position.

    Note:
        This function creates a new list and does not modify the input list in-place.
        All dictionaries in the input list must contain the keys specified in
        preserve_property_positions, or a KeyError will be raised.
    """

    # Create a dictionary to group items by their preserved properties
    groups = defaultdict(list)
    for index, item in enumerate(data):
        key = tuple(item[prop] for prop in preserve_property_positions)
        groups[key].append((index, item))
    
    # Create the result list with the same length as the input
    result = [None] * len(data)
    
    # Shuffle each group and place items back into the result list
    for group in groups.values():
        shuffled_group = group.copy()
        random.shuffle(shuffled_group)
        
        for (original_index, _), (_, shuffled_item) in zip(group, shuffled_group):
            result[original_index] = shuffled_item
    
    return result


def shuffle_blocked_trials(trial_blocks, indices, shuffle_function):
    """
    Shuffles the trials within selected blocks while preserving the overall block structure.
    
    Args:
        trial_blocks (list of lists): The list of trial blocks.
        indices (list of int): The indices of the blocks to be shuffled.
        shuffle_function (function): A function that shuffles a list.
        
    Returns:
        list of lists: The modified list of trial blocks.
    """
    # Concatenate the selected trial blocks
    selected_trials = [trial_blocks[i] for i in indices]
    combined_trials = [item for sublist in selected_trials for item in sublist]
    
    # Shuffle the combined trials
    result = shuffle_function(combined_trials)
    if result is not None:
        combined_trials = result
    
    # Split the shuffled trials back into the original block sizes
    block_sizes = [len(trial_blocks[i]) for i in indices]
    shuffled_blocks = []
    start = 0
    for size in block_sizes:
        shuffled_blocks.append(combined_trials[start:start+size])
        start += size
    
    # Insert the shuffled blocks back into the original list of trial blocks
    for i, idx in enumerate(indices):
        trial_blocks[idx] = shuffled_blocks[i]
    
    return trial_blocks