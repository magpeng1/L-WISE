import os
import argparse
import shutil
import numpy as np
import time

from lwise_psych_modules import *


# Function to generate balanced and shuffled blocks
def generate_balanced_blocks(num_conditions, num_trialsets, block_size, alternate=True):
  if block_size is None:
    block_size = num_conditions
  num_blocks = num_trialsets // block_size
  conditions_per_block = block_size // num_conditions
  blocks = []
  for _ in range(num_blocks):
    block = np.repeat(np.arange(num_conditions), conditions_per_block)
    if not alternate:
      np.random.shuffle(block)
    blocks.extend(block)
  return blocks


def main():
  parser = argparse.ArgumentParser(description='Create API Gateway API to call a lambda function to get trialset ID and session ID.')
  parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
  parser.add_argument('--experiment_number', type=int, required=True, help='Experiment number')
  parser.add_argument('--aws_prefix', type=str, help='Prefix for aws resources (e.g., to avoid confusion with other users)')
  parser.add_argument('--num_trialsets', type=int, required=True, help='Number of trialset .js files to generate. Be generous - e.g. 5x the intended number of participants')
  parser.add_argument('--delete_old_apis', action='store_true', help='Delete old API gateway APIs to avoid lots of duplicates (mostly for development purposes)')
  parser.add_argument('--fully_random_group_assignment', action='store_true', help='Each trialset has an independently randomly chosen condition group. Otherwise, groups are assigned with a blocked randomization design for balance purposes.')
  parser.add_argument('--num_conditions', type=int, default=None, help="Number of experimental conditions")
  parser.add_argument('--randomization_block_size', type=int, default=None, help="Size of blocks within which condition counts are balanced but still randomized.")
  parser.add_argument('--alternate_within_blocks', action='store_true', help="Alternate conditions within blocks (deterministic ordering).")
  parser.add_argument('--local_only', action='store_true', help="Don't do any of the AWS things, just generate trialsets locally")
  parser.add_argument('--files_only', action='store_true', help="Do nothing except re-generating trials and re-upload the files (no AWS config for lambda function, etc)")
  parser.add_argument('--aws_config_only', action='store_true', help="Do nothing except updating the AWS configuration (no file changes)")
  parser.add_argument('--completion_code', type=str, default=None, help='completion code (e.g. for Prolific)')
  parser.add_argument('--screen_out_code', type=str, default=None, help='Screenout code (e.g. for Prolific)')
  args = parser.parse_args()

  # Random seed init
  np.random.seed(int(time.time() % 1e6))

  # Assert that required files exist
  exp_id = f"{args.experiment_name}_{args.experiment_number}"
  exp_file = f"{exp_id}.html"
  config_file = "config.yaml"
  dataset_dirmap_file = "dataset_dirmap.csv"
  assert os.path.isfile(os.path.join("experiment_files", exp_id, exp_file)), \
    f"Main experiment file missing. Need: experiment_files/{exp_id}/{exp_file}"
  assert os.path.isfile(os.path.join("experiment_files", exp_id, config_file)), \
    f"Experiment config .yaml file missing. Need: experiment_files/{exp_id}/{config_file}"
  assert os.path.isfile(os.path.join("experiment_files", exp_id, dataset_dirmap_file)), \
    f"Dataset dirmap file missing. Need: experiment_files/{exp_id}/{dataset_dirmap_file}"


  # Assert that conditions and block sizes are specified in a way that makes sense
  assert args.fully_random_group_assignment or (args.num_conditions is not None and args.randomization_block_size is not None) or (args.num_conditions is not None and args.num_conditions == 1), \
    "Must either (a) specify --num_conditions 1, or (b) either use fully random group assignment or specify both num_conditions and randomization_block_size"
  if not args.fully_random_group_assignment and not args.num_conditions == 1:
    assert args.randomization_block_size % args.num_conditions == 0, \
      "Randomization block size must be a multiple of the number of conditions"
    assert args.num_trialsets % args.randomization_block_size == 0, \
      "Number of trialsets must be a multiple of the randomization block size"
    

  # Make local directory for deployment of this experiment and copy local files
  deploy_dir = os.path.join("deployed_experiments", exp_id)
  if not os.path.exists(deploy_dir):
    os.makedirs(deploy_dir)
  if not args.aws_config_only:
    shutil.copy(os.path.join("experiment_files", exp_id, exp_file), deploy_dir)
    shutil.copy(os.path.join("experiment_files", exp_id, config_file), deploy_dir)
    shutil.copy(os.path.join("experiment_files", exp_id, dataset_dirmap_file), deploy_dir)

  if not args.local_only:
    # Create S3 bucket for this experiment and upload experiment files
    # (dirmap and config only uploaded for record keeping purposes)
    bucket_name = f"{args.experiment_name.replace('_', '-').lower()}-{args.experiment_number}"
    if args.aws_prefix:
      bucket_name = f"{args.aws_prefix}-" + bucket_name
    if bucket_exists(bucket_name):
      print("WARNING: bucket already exists for this experiment. Skipping bucket creation")
      bucket_url = f"https://{bucket_name}.s3.amazonaws.com"
    else:
      bucket_url = create_s3_bucket(bucket_name, allow_public_files=True)
    if not args.aws_config_only:
      upload_s3_file(bucket_name, os.path.join(deploy_dir, exp_file), acl="public-read")
      upload_s3_file(bucket_name, os.path.join(deploy_dir, config_file))
      upload_s3_file(bucket_name, os.path.join(deploy_dir, dataset_dirmap_file), acl="public-read")

  if not args.aws_config_only:
    # Generate and upload trialsets (js files that each define a sequence of stimuli for one participant session)
    ts_dir = os.path.join(deploy_dir, "trialsets")
    if not os.path.exists(ts_dir):
      os.makedirs(ts_dir)

    # Determine condition assignment for each trialset
    if args.fully_random_group_assignment:
      condition_indices = np.random.randint(0, args.num_conditions, size=args.num_trialsets)
    else:
      condition_indices = generate_balanced_blocks(args.num_conditions, args.num_trialsets, args.randomization_block_size, alternate=args.alternate_within_blocks)
      print("TRIALSET CONDITION SEQUENCE:")
      print(condition_indices)

    gen_trials_func = generate_trials
    for ts_id, condition_idx in enumerate(condition_indices):
      ts_file = f"{exp_id}_trials_{ts_id}.js"
      gen_trials_func(os.path.join(deploy_dir, config_file), os.path.join(deploy_dir, dataset_dirmap_file), 
                      os.path.join(ts_dir, ts_file), condition_idx=int(condition_idx))
      if not args.local_only:
        upload_s3_file(bucket_name, os.path.join(ts_dir, ts_file), object_name=ts_file, loc_in_bucket="trialsets", acl="public-read")

  if not args.local_only and not args.files_only:
    # Deploy lambda function for getting session metadata (broadly includes API, IAM role, and dynamodb tables for storing metadata)
    print("Creating DynamoDB tables for session metadata lambda function...")
    counter_table_name, mapper_table_name = create_dynamodb_tables(args.experiment_name, args.experiment_number, args.aws_prefix)
    if args.aws_prefix: 
      session_metadata_lambda_name = f"{exp_id}_session_metadata"
    if args.aws_prefix: 
      session_metadata_lambda_name = f"{args.aws_prefix}_" + session_metadata_lambda_name

    lambda_env_vars = {
      "EXPERIMENT_NAME": args.experiment_name,
      "EXPERIMENT_NUMBER": str(args.experiment_number),
      "AWS_PREFIX": args.aws_prefix,
      "BUCKET_NAME": bucket_name,
      "COUNTER_TABLE_NAME": counter_table_name,
      "MAPPING_TABLE_NAME": mapper_table_name,
    }
    if args.completion_code is not None:
      lambda_env_vars['COMPLETION_CODE'] = args.completion_code
    if args.screen_out_code is not None:
          lambda_env_vars['SCREEN_OUT_CODE'] = args.screen_out_code

    session_metadata_api_url = deploy_session_metadata_lambda_function(session_metadata_lambda_name, bucket_url, delete_old_apis=args.delete_old_apis, env_vars=lambda_env_vars)

  
  # Create and upload aws constants file
  if not args.local_only and not args.files_only:
    aws_constants = {"session_metadata_api_url": '"' + session_metadata_api_url + '"'}
    write_js_vars(os.path.join(deploy_dir, "aws_constants.js"), aws_constants, "let")
    upload_s3_file(bucket_name, os.path.join(deploy_dir, "aws_constants.js"), acl="public-read")
    print("Uploaded aws_constants.js:", aws_constants)

  if not args.local_only:
    print(f"Experiment {exp_id} successfully deployed at: {os.path.join(bucket_url, exp_file)}")
    if not args.files_only:
      print(f"URL to use for in-person experiments: {os.path.join(bucket_url, exp_file)}?PLATFORM=inperson&trialsubmit={session_metadata_api_url}&sessionsubmit={session_metadata_api_url}")
      print(f"   (Note: you can change PLATFORM=inperson to PLATFORM=anythingyoulike, the platform will just be recorded as such in the data)")
      print(f"URL to use on Prolific: {os.path.join(bucket_url, exp_file)}?PID={{{{%PROLIFIC_PID%}}}}&STUDY_ID={{{{%STUDY_ID%}}}}&SESSION_ID={{{{%SESSION_ID%}}}}&PLATFORM=prolific&trialsubmit={session_metadata_api_url}&sessionsubmit={session_metadata_api_url}")
      print(f"URL to use on Mechanical Turk: {os.path.join(bucket_url, exp_file)}?PLATFORM=mturk&trialsubmit={session_metadata_api_url}&sessionsubmit={session_metadata_api_url}")
      print(f"URL to test experiment interface in browser: {os.path.join(bucket_url, exp_file)}?TRIALSET_ID=0&trialsubmit={session_metadata_api_url}&sessionsubmit={session_metadata_api_url}&PLATFORM=test")
      print("    (To test with a different trialset_id (e.g., trialset 42), replace TRIALSET_ID=0 with TRIALSET_ID=42 in the URL above)")
  else:
    print("Experiment files successfully generated")

if __name__ == "__main__":
  main()
