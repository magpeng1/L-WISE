import os
import pandas as pd
import argparse
import random
import json
from .generate_trials import generate_trial
from .generate_trials_helpers import *
from .utils import write_js_vars, load_configs


def generate_prespecified_trials(config_path, trial_spec_path, output_js_dir, experiment_name, experiment_number, write_csvs=True):

  trial_config, _, _ = load_configs(config_path)

  if not os.path.exists(output_js_dir) and len(output_js_dir) > 0:
    os.makedirs(os.path.dirname(output_js_dir))

  # Read the trial specification from CSV
  all_trials_df = pd.read_csv(trial_spec_path)

  classes = list(all_trials_df["class"].unique())

  trialset_ids = sorted(list(all_trials_df["trialset_id"].unique()))
  loaded_session_dfs = [all_trials_df[all_trials_df["trialset_id"] == tid] for tid in trialset_ids]

  for idx, session_df in enumerate(loaded_session_dfs):
    trialset_id = trialset_ids[idx]

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

    session_trials = []
    for _, row in session_df.iterrows():
      block_config = trial_config.copy()
      row_dict = row.to_dict()
      for key in row_dict:
        block_config[key] = row_dict[key]
      choice_names, choice_urls = get_choice_names_and_urls(block_config, trial_config, class_to_url_dict)
      trial = generate_trial(row, block_config, trial_config, choice_urls, choice_names, class_to_url_dict)
      session_trials.append(trial)

    session_trials_df = pd.DataFrame(session_trials)

    # Write to a JS file
    trialset_output_js = os.path.join(output_js_dir,experiment_name + "_"  + str(experiment_number) + "_trials_" + str(trialset_id) + ".js")
    write_js_vars(trialset_output_js,  {"trial_variables": json.dumps(session_trials, indent=2)}, "let")

    # Write a .csv to the same location
    if write_csvs:
      session_trials_df.to_csv(os.path.splitext(trialset_output_js)[0] + ".csv")


def main():
  parser = argparse.ArgumentParser(description='Generate trial data for a JS experiment.')
  parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
  parser.add_argument('--experiment_number', type=int, required=True, help='Experiment number')
  parser.add_argument('--config_path', type=str, default="exp_config.yaml", help='Path to the configuration file')
  parser.add_argument('--trial_spec_path', type=str, default="trial_spec.csv", help='Path to the CSV file indicating which participant will view which images')
  parser.add_argument('--output_js_dir', type=str, default="bin", help='Path for the output JS files')
  
  args = parser.parse_args()

  generate_prespecified_trials(args.config_path, args.trial_spec_path, args.output_js_dir, args.experiment_name, args.experiment_number)


if __name__ == "__main__":
  main()
