import os
import pandas as pd
import argparse
import random
import json
import time
from collections import deque
import itertools
import copy
from .generate_trials_helpers import *
from .utils import write_js_vars, load_configs


def make_trial_dict(entry, i_correct_choice, choice_image_urls, choice_names, trial_config):
  trial_dict = {
    "trial_type": entry['trial_type'],
    "class": entry["class"] if "class" in entry else choice_names[i_correct_choice].replace(" ", "_"),
    "split": entry['split'] if 'split' in entry else 'na',
    "i_correct_choice": i_correct_choice,
    "stimulus_image_url": entry['url'],
    **trial_config, # Unpacks everything in trial_config into the dictionary
    "choice_names": choice_names,
    "choice_image_urls": choice_image_urls,
  }

  # if entry contains a stimulus_duration, override the one in trial_config
  if "stimulus_duration" in entry and entry["stimulus_duration"]:
    trial_dict["stimulus_duration_msec"] = entry["stimulus_duration"]

  if "difficulty" in entry:
    trial_dict["difficulty"] = entry["difficulty"]

  return trial_dict


def generate_trial(entry, block_config, trial_config, choice_image_urls, choice_names, class_to_url_dict, trial_types_dict=None, choice_name_to_bucket_dict=None):
  # Note about propagation of settings: Settings within a trial_type always take priority, followed by block-specific settings, followed by default settings in trial_config.
  # So if you specify "stimulus_duration_msec" in both the block config and the trial config, the one in the block config will be used. 
  # If you additionally specify stimulus_duration_msec as part of a trial_type within that same block, the trial_type value will be used. 
  # If you have another block with no specification of stimulus_duration_msec, the value in trial_config will be used by default. 

  if "i_correct_choice" in entry:
    i_correct_choice = entry["i_correct_choice"]
  else:
    i_correct_choice = choice_image_urls.index(class_to_url_dict[entry['class']])

  if choice_name_to_bucket_dict:
    entry['url'] = replace_bucket_name_in_url(entry['url'], choice_name_to_bucket_dict[choice_names[i_correct_choice]])

  # Override trial_config vars with block_config vars if there is overlap
  trial_config = copy.deepcopy(trial_config)
  trial_config['calibration'] = 0

  # Override these variables even if they don't already exist in trial_config:
  block_vars_to_trials = ["block", "condition_idx"]
  for bvar in block_vars_to_trials:
    if bvar in block_config:
      trial_config[bvar] = block_config[bvar]

  # For all cases where a key exists in both trial_config and block_config, overwrite the trial_config one from block_config
  for key in block_config.keys():
    if key in trial_config or key in ["show_test_instructions"]:
      trial_config[key] = block_config[key]

  if trial_types_dict and 'trial_type' in entry and entry['trial_type'] in trial_types_dict:

    if "bucket" in trial_types_dict[entry['trial_type']]:
      entry['url'] = replace_bucket_name_in_url(entry['url'], trial_types_dict[entry['trial_type']]['bucket'])
    
    for trial_type_attribute in trial_types_dict[entry['trial_type']].keys():
      if trial_type_attribute in ["bucket"]:
        continue
      else:
        trial_config[trial_type_attribute] = trial_types_dict[entry['trial_type']][trial_type_attribute]

  if "trial_type" not in entry:
    entry["trial_type"] = "new_stimulus"

  trial = make_trial_dict(entry, i_correct_choice, choice_image_urls, choice_names, trial_config)
  return trial


def generate_calibration_trial(block_config, trial_config, choice_image_urls, choice_names, cstim, choice_url_suffix="_icon.png"):

  # Override trial_config vars with block_config vars if there is overlap
  trial_config = copy.deepcopy(trial_config)
  trial_config['calibration'] = 1
  if "block" in block_config:
    trial_config["block"] = block_config["block"]
  for key in block_config.keys():
    if key in trial_config or key in ["show_test_instructions"]:
      trial_config[key] = block_config[key]

  cstims = trial_config["calibration_stimuli"]

  if cstim not in cstims:
    raise NotImplementedError(f'generate_calibration_trials is currently only implemented for the following entry in config.yaml: calibration_stimuli: {cstim}')
  assert len(cstims) == 2, f"generate_calibration_trial is only implemented for 2 different calibration stimuli. This list can only have 2 items: {cstims}"

  if "choice_image_bucket_name" in trial_config:
    choice_image_bucket_name = trial_config["choice_image_bucket_name"]
  else:
    choice_image_bucket_name = "easy-imagenet-media" # TEMP: for backwards-compatibility with early experiments

  if "keypress_fj_response" in trial_config and trial_config["keypress_fj_response"]:
    kps = ["_F", "_J"]
  else:
    kps = [""] * len(choice_image_urls)

  # Replace two random labels with cstims[0] and cstims[1] labels
  modified_choice_urls = choice_image_urls.copy()
  modified_choice_names = choice_names.copy()
  random_inds = random.sample(range(len(modified_choice_urls)), 2)
  modified_choice_urls[random_inds[0]] = f"https://{choice_image_bucket_name}.s3.amazonaws.com/{cstim}{kps[random_inds[0]]}{choice_url_suffix}"
  modified_choice_urls[random_inds[1]] = f"https://{choice_image_bucket_name}.s3.amazonaws.com/{cstims[0] if cstim == cstims[1] else cstims[1]}{kps[random_inds[1]]}{choice_url_suffix}"
  modified_choice_names[random_inds[0]] = cstim
  modified_choice_names[random_inds[1]] = cstims[0] if cstim == cstims[1] else cstims[1]
  i_correct_choice = random_inds[0]
  trial = make_trial_dict({
      "url": f"https://{choice_image_bucket_name}.s3.amazonaws.com/{cstim}.png",
      "class": cstim,
      "split": None,
      "trial_type": "calibration"
    }, i_correct_choice, modified_choice_urls, modified_choice_names, trial_config)
  return trial


def generate_trials(config_path, dirmap_path, output_js_path, write_csvs=True, condition_idx=None):

  trial_config, session_config, _ = load_configs(config_path)

  if condition_idx is not None:
    trial_config["condition_idx"] = condition_idx

  if not os.path.exists(os.path.dirname(output_js_path)) and len(os.path.dirname(output_js_path)) > 0:
    os.makedirs(os.path.dirname(output_js_path))

  # Iterate through the dataframe and add images to the trial set
  dataset_df = pd.read_csv(dirmap_path)

  # Check if we are using a difficulty curriculum
  difficulty_curriculum = False
  for block_ind, block_config in enumerate(session_config):
    if ("difficulty_curriculum" in trial_config and trial_config["difficulty_curriculum"]) or \
      ("difficulty_curriculum" in block_config and block_config["difficulty_curriculum"]) or \
      ("conditional_trial_types" in block_config and "difficulty_curriculum" in block_config["conditional_trial_types"][trial_config["condition_idx"]][next(iter(block_config["conditional_trial_types"][trial_config["condition_idx"]]))] and block_config["conditional_trial_types"][trial_config["condition_idx"]][next(iter(block_config["conditional_trial_types"][trial_config["condition_idx"]]))]["difficulty_curriculum"]):
      difficulty_curriculum = True

  if difficulty_curriculum:
    assert 'difficulty' in dataset_df.columns, "Dataset must have a 'difficulty' column when using difficulty curriculum"
    assert trial_config['curriculum_num_blocks'] is not None, "Must specify number of easy blocks for difficulty curriculum"
    assert trial_config['curriculum_num_blocks'] > 0, "Must specify more than 0 curriculum blocks if using difficulty curriculum"

    # Sort dataset by difficulty within each class
    dataset_df = dataset_df.sort_values(['class', 'difficulty'])

    # Sample trials for the non-curriculum part of the session (i.e., the test session plus any test-like training blocks at the end)
    orig_len = len(dataset_df)
    hard_trials_df = sample_non_curriculum_trials(dataset_df, session_config[trial_config['curriculum_num_blocks']:])
    dataset_df = dataset_df[~dataset_df.index.isin(hard_trials_df.index)]
    assert len(hard_trials_df) + len(dataset_df) == orig_len

  if "choice_names_order_shuffle" in trial_config and trial_config["choice_names_order_shuffle"]:
    random.shuffle(trial_config["choice_names_order"])

  # If specified, randomly shuffle the choice name order and aliases for the whole session
  if "choice_aliases_random_shuffle" in trial_config and trial_config["choice_aliases_random_shuffle"]:
    choice_names = trial_config["choice_names_order"]

    if "choice_names_aliases" in trial_config: 
      choice_names_aliases = trial_config["choice_names_aliases"]
      assert len(choice_names_aliases) == len(choice_names), f"If you set choice_aliases_random_shuffle=true in the trial_config, you must have either no aliases or 1 alias for every choice name. Choice names: {choice_names}. Aliases: {choice_name_aliases}"
      
      # Keep track of the positions in the choice names order of the original aliases
      original_aliases_values = [choice_names_aliases[name] for name in choice_names]

      # Randomly shuffle the choice names
      random.shuffle(choice_names)

      # Now, reassign choice_names_aliases based on the new order of choice_names
      choice_names_aliases_shuffled = dict(zip(choice_names, original_aliases_values))
      trial_config["choice_names_aliases"] = choice_names_aliases_shuffled
    else:
      random.shuffle(choice_names)

    trial_config["choice_names_order"] = choice_names

  classes = list(dataset_df["class"].unique())

  n_repeats_total = 0
  for block_ind, block_config in enumerate(session_config):
    if "n_repeats_of_one_stim" in block_config:
      n_repeats_total = n_repeats_total + block_config["n_repeats_of_one_stim"]
    
  trial_block_lists = []
  for block_ind, block_config in enumerate(session_config):

    block_config["block"] = block_ind

    if "conditional_trial_types" in block_config:
      assert "trial_types" not in block_config, "If specifying conditional_trial_types, do not also separately specify trial_types - choose one or the other."
      block_config["trial_types"] = block_config["conditional_trial_types"][condition_idx]

    trial_types = block_config.get("trial_types", None)
    if trial_types:
      dataset_df["trial_type"] = None

    if "show_test_instructions" not in block_config:
      block_config["show_test_instructions"] = False

    n_per_class_total = sum([block_config["n_trials_per_class_" + s] for s in ["train", "val", "test"]])
    sess_df = pd.DataFrame()
    if len(classes) * n_per_class_total <= len(dataset_df):
      sampled_indices = set()
      for hclass in classes:
        for split in ["train", "val", "test"]:
          class_split_df = dataset_df[(dataset_df["split"] == split) & (dataset_df["class"] == hclass)]
          sample_n = block_config["n_trials_per_class_" + split]

          # If multiple trial types are provided, evenly distribute them among the trials
          ## NOTE!! This approach currently does NOT support difficulty-based curriculum ordering
          if trial_types:
            # Calculate the number of trials per type
            trial_type_names = list(trial_types.keys())
            random.shuffle(trial_type_names)
            n_trial_types = len(trial_type_names)
            n_per_trial_type = int(sample_n / n_trial_types)
            remainder = sample_n % n_trial_types

            split_sample_dfs = []
            for trial_type_name in trial_type_names:
              # Determine the number of trials for this trial_type, adding one if there's a remainder
              n_trials_this_trial_type = n_per_trial_type + (1 if remainder > 0 else 0)
              if remainder > 0:
                remainder -= 1

              # Sample n_trials_this_trial_type trials with the current trial type
              if n_trials_this_trial_type > 0:
                if difficulty_curriculum and block_ind < trial_config['curriculum_num_blocks']:
                  sampled_df = difficulty_curriculum_sample_block(class_split_df, n_trials_this_trial_type, trial_config, session_config, block_ind, block_config)
                else:
                  sampled_df = class_split_df.sample(n=n_trials_this_trial_type)
                sampled_df["trial_type"] = trial_type_name
                split_sample_dfs.append(sampled_df)

                # Remove sampled entries to avoid re-selection
                class_split_df = class_split_df.drop(sampled_df.index)
                sampled_indices.update(sampled_df.index)

            # Concatenate all sampled dfs for this class and split
            if split_sample_dfs:
              sampled_df = pd.concat(split_sample_dfs).reset_index(drop=True)
            else:
              sampled_df = None

          else:
            # Sample without considering trial types
            if difficulty_curriculum and block_ind < trial_config['curriculum_num_blocks']:
              sampled_df = difficulty_curriculum_sample_block(class_split_df, sample_n, trial_config, session_config, block_ind, block_config)
            else:
              sampled_df = class_split_df.sample(n=sample_n)
            sampled_indices.update(sampled_df.index)

          if sampled_df is not None and len(sampled_df) > 0:
            sess_df = pd.concat([sess_df, sampled_df])

    else:
      # sess_df = dataset_df
      raise ValueError("Value of n_per_class_total is too high for the size of the dataset.")
    
    # If in the first block, sample one extra row (WITH STIMS FROM ***VAL*** SET) to use as a repeat trial
    # (unless split is otherwise specified in trial_config)
    if n_repeats_total > 0 and block_ind == 0:
      dataset_df_copy = dataset_df.copy()
      dataset_df_copy = dataset_df_copy.drop(index=sampled_indices)
      if 'repeat_stimulus_split' in trial_config:
        repeat_stimulus_split = trial_config['repeat_stimulus_split']
      else:
        repeat_stimulus_split = "val"
      split_df = dataset_df_copy[(dataset_df_copy["split"] == repeat_stimulus_split)]
      r_sampled_df = split_df.sample(n=1, random_state=int(time.time() % 1e6))
      r_sampled_df["trial_type"] = "repeat_stimulus"
      sampled_indices.update(r_sampled_df.index)

    # Add repeat trials
    if "n_repeats_of_one_stim" in block_config and block_config["n_repeats_of_one_stim"] > 0:
      repeat_df = pd.concat([r_sampled_df.iloc[0].to_frame().T] * block_config["n_repeats_of_one_stim"], ignore_index=True)
    else:
      repeat_df = None
    
    # Remove sampled rows from dataset_df to avoid resampling
    # Only do this if we are NOT doing curriculum
    # If we are doing curriculum and we are in the last curriculum block, we reassign dataset_df to the hard_trials_df we set aside at the beginning
    if difficulty_curriculum and block_ind < trial_config['curriculum_num_blocks']:
      if block_ind == trial_config['curriculum_num_blocks'] - 1: # If we're in the final curriculum block
        dataset_df = hard_trials_df # Reassign the trials we set aside at the beginning
    else:
      dataset_df = dataset_df.drop(index=sampled_indices)

    if "choice_url_suffix" in trial_config:
      choice_url_suffix = trial_config["choice_url_suffix"]
    else:
      choice_url_suffix = "_icon.png"

    if "choice_names_aliases" in trial_config and trial_config["choice_names_aliases"]:
      choice_name_aliases_partial = trial_config["choice_names_aliases"]
    else:
      choice_name_aliases_partial = {}
    
    # Fill in any missing items in the dict (if it's empty, every class name just maps to itself - same for any classes that don't have aliases in the dict)
    # Note: the default "choice name" is just the class name with any underscores replaced by spaces
    all_choice_names = [cl.replace('_', ' ') for cl in classes]
    choice_name_aliases = {ch: (choice_name_aliases_partial[ch] if ch in choice_name_aliases_partial else ch) for ch in all_choice_names}
    class_to_url_dict = {cl: f"https://{trial_config['choice_image_bucket_name']}.s3.amazonaws.com/{choice_name_aliases[cl.replace('_', ' ')].replace(' ', '_')}{choice_url_suffix}" for cl in classes}

    # If n_classes_per_bucket is specified, we assign each class to different buckets accordingly. 
    if "n_classes_per_bucket" in block_config:

      # If multiple dicts specified, randomly choose one.
      if isinstance(block_config["n_classes_per_bucket"], list):
        if condition_idx is not None:
          n_classes_per_bucket = block_config["n_classes_per_bucket"][condition_idx]
        else:
          n_classes_per_bucket = block_config["n_classes_per_bucket"][random.randint(0, len(block_config["n_classes_per_bucket"])-1)]
      else:
        n_classes_per_bucket = block_config["n_classes_per_bucket"]

      # Assert the length of choice_names matches the sum of values in the dict
      expected_length = sum(n_classes_per_bucket.values())
      choice_names, _ = get_choice_names_and_urls(block_config, trial_config, class_to_url_dict)
      assert len(choice_names) == expected_length, f"Length of choice_names ({len(choice_names)}) does not match expected total ({expected_length})."

      # Shuffle the choice_names to ensure randomness in selection
      choice_names_shuffled = copy.deepcopy(choice_names)
      random.shuffle(choice_names_shuffled)

      choice_name_to_bucket_dict = {}
      for bucket_name, num_choices in n_classes_per_bucket.items():
        selected_choices = [choice_names_shuffled.pop() for _ in range(num_choices)]
        for choice in selected_choices:
          choice_name_to_bucket_dict[choice] = bucket_name
    else:
      choice_name_to_bucket_dict = None

    # Generate main trials
    main_trials = []
    for _, row in sess_df.iterrows():
      choice_names, choice_urls = get_choice_names_and_urls(block_config, trial_config, class_to_url_dict)
      trial = generate_trial(row, block_config, trial_config, choice_urls, choice_names, class_to_url_dict, trial_types_dict=trial_types, choice_name_to_bucket_dict=choice_name_to_bucket_dict)
      main_trials.append(trial)

    # Generate repeat trials
    repeat_trials = []
    if repeat_df is not None:
      for _, row in repeat_df.iterrows():
        trial = generate_trial(row, block_config, trial_config, choice_urls, choice_names, class_to_url_dict, trial_types_dict=trial_types, choice_name_to_bucket_dict=choice_name_to_bucket_dict)
        repeat_trials.append(trial)

    # Generate calibration trials
    calibration_stims = trial_config["calibration_stimuli"] * (block_config["n_calibration_trials"] // len(trial_config["calibration_stimuli"]))
    calibration_trials = []
    for stim in calibration_stims:
      choice_names, choice_urls = get_choice_names_and_urls(block_config, trial_config, class_to_url_dict)
      calibration_trials.append(generate_calibration_trial(block_config, trial_config, choice_urls, choice_names, stim, choice_url_suffix=choice_url_suffix))

    # Intersperse repeat and calibration trials with main trials
    total_block_trials = main_trials + repeat_trials + calibration_trials
    random.shuffle(total_block_trials)

    # The code below is an alternate way to shuffle to preserve certain ordering rules for curriculum - but it may still create biases due to the way trials are split up among blocks (e.g., ordering within a split may not matter, but within a class it might)
    # if 'difficulty_curriculum' in trial_config and trial_config['difficulty_curriculum']:
    #   total_block_trials = main_trials
    #   other_trials = repeat_trials + calibration_trials
    #   random.shuffle(other_trials)
    #   for trial in other_trials:
    #     insert_position = random.randint(0, len(total_block_trials))
    #     total_block_trials.insert(insert_position, trial)

    trial_block_lists.append(total_block_trials)

  # Shuffle the trials, if appropriate
  idx_of_blocks_to_combine_and_shuffle = []
  any_combine_and_shuffle = False
  for block_idx, block in enumerate(trial_block_lists):
    combine_and_shuffle = None
    for trial in block:
       if 'combine_and_shuffle' in trial:
        if combine_and_shuffle is None:
            combine_and_shuffle = trial['combine_and_shuffle']
        elif combine_and_shuffle != trial['combine_and_shuffle']:
            raise ValueError("All trials within a block must have the same 'combine_and_shuffle' value.")
    if combine_and_shuffle:
      idx_of_blocks_to_combine_and_shuffle.append(block_idx)
      any_combine_and_shuffle = True
  
  if any_combine_and_shuffle: # Note: see documentation for constrained_shuffle in generate_trials_helpers.py
    trial_block_lists = shuffle_blocked_trials(trial_block_lists, idx_of_blocks_to_combine_and_shuffle, constrained_shuffle)

  # Combine total trials
  total_trials = list(itertools.chain(*trial_block_lists))
  total_trials_df = pd.DataFrame(total_trials)

  ## ASSERTIONS TO MAKE SURE THERE ARE NO DUPLICATES WHERE THERE AREN'T SUPPOSED TO BE ##

  total_trials_df_copy = total_trials_df.copy(deep=True)

  # Extract the file names from stimulus_image_url and store in a new column
  total_trials_df_copy['stimulus_filename'] = total_trials_df_copy['stimulus_image_url'].apply(lambda x: x.split('/')[-1])

  # Assert that each file name has an extension
  assert all(total_trials_df_copy['stimulus_filename'].apply(lambda x: '.' in x)), "One or more stimulus_image_urls do not end with a file name and extension"

  # Ensure that the required columns exist, defaulting to None or False if they don't
  if 'trial_type' not in total_trials_df_copy.columns:
    total_trials_df_copy['trial_type'] = None
  if 'difficulty_curriculum' not in total_trials_df_copy.columns:
    total_trials_df_copy['difficulty_curriculum'] = False

  total_trials_df_copy['difficulty_curriculum'] = total_trials_df_copy['difficulty_curriculum'].astype('bool')

  # Create a subset of the dataframe without the exceptions
  df_no_exceptions = total_trials_df_copy[
    (~total_trials_df_copy['trial_type'].isin(['repeat_stimulus', 'calibration'])) & 
    (~total_trials_df_copy['difficulty_curriculum'])
  ]

  # Assert that there are no duplicate stimulus_image_urls among the non-exception rows
  assert df_no_exceptions['stimulus_image_url'].is_unique, "Duplicate stimulus_image_url found among non-exception trials"

  # Assert that there are no duplicate stimulus file names among the non-exception rows (stricter version of the no-duplicate-urls assertion)
  assert df_no_exceptions['stimulus_filename'].is_unique, "Duplicate stimulus filenames found among non-exception trials"
  
  ########################### END OF ASSERTIONS #######################################

  # Write to a JS file
  write_js_vars(output_js_path, {"trial_variables": json.dumps(total_trials, indent=2)}, "let")

  # Write a .csv to the same location
  if write_csvs:
    total_trials_df.to_csv(os.path.splitext(output_js_path)[0] + ".csv")


def main():
  parser = argparse.ArgumentParser(description='Generate trial data for a JS experiment.')
  parser.add_argument('--config_path', type=str, default="exp_config.yaml", help='Path to the configuration file')
  parser.add_argument('--dirmap_path', type=str, default="dirmap.csv", help='Path to the directory map CSV')
  parser.add_argument('--output_js_path', type=str, default="bin/trials.js", help='Path for the output JS file')
  
  args = parser.parse_args()

  generate_trials(args.config_path, args.dirmap_path, args.output_js_path, args.n_calibration_trials)


if __name__ == "__main__":
  main()
