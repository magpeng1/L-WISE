import os
import numpy as np
import boto3
import json
import argparse
import traceback
from tqdm import tqdm
import xarray as xr
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import unquote, urlparse

# below up to 'from lwise_psych_modules' is new to access parent dir
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

REPO_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_DIR / "experiment_files"
DEPLOY_ROOT     = REPO_DIR / "deployed_experiments"

from lwise_psych_modules.process_session_data import process_session_data, get_bonus_amount_from_message

def fetch_experiment_data(dynamodb_client, table_name):
    """Fetch all data entries from the DynamoDB table."""
    items = []
    scan_kwargs = {
        'TableName': table_name,
    }

    done = False
    start_key = None
    while not done:
        if start_key:
            scan_kwargs['ExclusiveStartKey'] = start_key
        response = dynamodb_client.scan(**scan_kwargs)
        items.extend(response.get('Items', []))
        start_key = response.get('LastEvaluatedKey', None)
        done = start_key is None

    return items

def fetch_json_from_s3(s3_client, url):
    """Fetch JSON data from S3 given a URL."""
    parsed_url = urlparse(url)
    bucket = parsed_url.netloc.split('.')[0]
    key = unquote(parsed_url.path).lstrip('/')
    
    response = s3_client.get_object(Bucket=bucket, Key=key)
    content = response['Body'].read().decode('utf-8')
    return json.loads(content)


def fetch_partial_trials(s3_client, bucket, assignment_id):
    """Fetch partial trial data from S3 for a given assignment ID."""
    s3 = boto3.resource('s3')
    bucket_resource = s3.Bucket(bucket)
    
    prefix = f"trial_data/{assignment_id}/"
    
    # Use paginator for efficient listing
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    trial_files = []
    for page in pages:
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.json'):
                trial_files.append(obj['Key'])
    
    trial_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    trial_idxs_outer = [int(x.split('_')[-1].split('.')[0]) for x in trial_files]
    try:
        max_trial = int(trial_files[-1].split('_')[-1].split('.')[0])
    except IndexError:
        max_trial = -1
    
    def download_file(file):
        obj = bucket_resource.Object(file)
        response = obj.get()
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    
    trials = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_file = {executor.submit(download_file, file): file for file in trial_files}
        for future in as_completed(future_to_file):
            trials.append(future.result())
    
    trial_idxs_inner = []
    for trial in trials:
        assert trial['assignment_id'] == assignment_id
        trial_idxs_inner.append(trial['trial_index'])

    assert set(trial_idxs_inner) == set(trial_idxs_outer), "The lists do not contain the same elements"
    assert len(trial_idxs_inner) == len(set(trial_idxs_inner)), "trial_idxs_inner contains duplicates"
    assert len(trial_idxs_outer) == len(set(trial_idxs_outer)), "trial_idxs_outer contains duplicates"

    return trials, max_trial

def check_for_full_test_in_partial_trials(ptrials, test_blocks=[8,9], test_trial_minimum=40):
    test_trial_count = 0
    test_blocks = [str(b) for b in test_blocks]
    for trial in ptrials:
        if "trial_outcome" in trial and "block" in trial["trial_outcome"] and str(trial["trial_outcome"]["block"]) in test_blocks and not trial["trial_outcome"]["trial_type"] in ["calibration", "repeat_stimulus"]:
            test_trial_count = test_trial_count + 1

    if test_trial_count >= test_trial_minimum:
        print(test_trial_count)
        return True
    else:
        print(f"This participant with partial trials only had {test_trial_count} test trials. Excluding their data.")
        return False

def process_items(items, experiment_name, experiment_number, aws_prefix, s3_client, get_partial_trials, get_all_partial_trials, test_trial_minimum, test_blocks, check_if_was_screened_out):
    """Process items and combine into a single xarray dataset."""
    datasets = []
    complete_not_logged = []  # Used to log any completed assignments that didn't have full data saved for some reason
    bonuses = {}
    for item in tqdm(items):
        session_data = None
        enough_data = False
        if 'session_data_url' in item and (not check_if_was_screened_out or ('was_screened_out' in item and item['was_screened_out']['BOOL'] == False)):
            session_data_url = item['session_data_url']['S']
            session_data = fetch_json_from_s3(s3_client, session_data_url)['session_data']
            enough_data = True
        elif 'session_data_url_auto' in item and ('was_screened_out' in item and item['was_screened_out']['BOOL'] == False):
            print(f"Using automatically stored version of data for participant {item['worker_id']['S']}")
            session_data_url = item['session_data_url_auto']['S']
            session_data = fetch_json_from_s3(s3_client, session_data_url)['session_data']
            enough_data = True
        if get_all_partial_trials or ('session_data_url' not in item and 'session_data_url_auto' not in item and get_partial_trials and ('was_screened_out' in item and item['was_screened_out']['BOOL'] == False)):
            bucket = f"{experiment_name.replace('_', '-').lower()}-{experiment_number}"
            if aws_prefix:
                bucket = aws_prefix + "-" + bucket
            assignment_id = item['assignment_id']['S']
            try:
                session_data, max_trial = fetch_partial_trials(s3_client, bucket, assignment_id)
            except Exception as e:
                print(f"Error while fetching partial trials: {str(e)}")
                print(traceback.format_exc())

            if session_data and (session_data[-1]['trial_type'] == "survey-text" or session_data[-1]['trial_index'] > 850 or max_trial > 850): ## TEMPORARY MEASURE designed specifically for ham4_learn_4 experiment which had data logging issues for some participants
                complete_not_logged.append(item['worker_id']['S'])

            if session_data and test_trial_minimum is not None and test_trial_minimum > 0:
                if check_for_full_test_in_partial_trials(session_data, test_blocks=test_blocks, test_trial_minimum=test_trial_minimum):
                    print(f"Complete test results recovered from partial trials of participant {item['worker_id']['S']}!")
                    enough_data = True
                else:
                    enough_data = False

        if 'bonus_usd' in item:
            bonus_usd = round(float(item['bonus_usd']['S']), 2)
        else:
            bonus_usd = None
        if bonus_usd is None and session_data:
            try:
                bonus_usd = get_bonus_amount_from_message({'trials': session_data})
            except (KeyError, IndexError):
                bonus_usd = None
            except ValueError as e:
                print(e)
                print("Look for the bonus message in here: ", session_data[-4:])
                bonus_usd = None
        
        if bonus_usd:
            bonuses[item['worker_id']['S']] = bonus_usd
        
        if session_data and enough_data:
            ds = process_session_data(session_data)
            if ds is not None:
                # Some numerical variables end up having mixed types. Set them all to float
                for mixed_type_var in ['perf', 'reaction_time_msec']:
                    if mixed_type_var in ds:
                        ds[mixed_type_var] = ds[mixed_type_var].astype(np.float64)

                ds = ds.assign_coords(
                    assignment_id=item['assignment_id']['S'],
                    worker_id=item['worker_id']['S'] if 'worker_id' in item else 'UNKNOWN',
                    trialset_id=int(item['trialset_id']['N']),
                    platform=item.get('platform', {}).get('S', 'UNKNOWN'),
                    bonus_usd=float(item.get('bonus_usd', {}).get('S', '0'))
                )
                datasets.append(ds)

    print("BONUSES:")
    for pt in bonuses.keys():
        print(pt + "," + str(bonuses[pt]))
    print("------End of bonuses------")

    if len(complete_not_logged) > 0:
        print("The following participants appear to have completed the full task, but their full data was not recorded for some reason")
        for wid in complete_not_logged:
            print(wid)

    if datasets:
        print(f"{len(datasets)} session JSONs processed")
        combined_ds = xr.concat(datasets, dim='participant')
        return combined_ds
    else:
        return None

def main(experiment_name, experiment_number, aws_prefix, save_dir, get_partial_trials, get_all_partial_trials, require_min_test_trials, test_blocks, check_if_was_screened_out):
    # Initialize DynamoDB and S3 clients
    dynamodb_client = boto3.client('dynamodb', region_name='us-east-2')
    s3_client = boto3.client('s3', region_name='us-east-2')

    # Construct the table name
    table_name = f"{aws_prefix}_{experiment_name}_{experiment_number}_trialset_id_mapper"

    print(f"Fetching data from table: {table_name}")

    try:
        items = fetch_experiment_data(dynamodb_client, table_name)
        print(f"Retrieved {len(items)} items from DynamoDB")

        combined_ds = process_items(items, experiment_name, experiment_number, aws_prefix, s3_client, get_partial_trials, get_all_partial_trials, require_min_test_trials, test_blocks, check_if_was_screened_out)

        if combined_ds is not None:
            # Save the combined dataset
            save_path = os.path.join(save_dir, f"{experiment_name}_{experiment_number}_combined_dataset.h5")
            print(f"Saving combined dataset to: {save_path}")

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Check if file exists and remove it
            if os.path.exists(save_path):
                print(f"Existing file found. Removing: {save_path}")
                os.remove(save_path)

            combined_ds.to_netcdf(path=save_path)
            print("Data saved successfully")
        else:
            print("No valid data to save")

    except ClientError as e:
        print(f"Error accessing DynamoDB or S3: {e.response['Error']['Message']}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and combine experiment data from DynamoDB and S3")
    parser.add_argument('--experiment_name', required=True, help="Name of the experiment")
    parser.add_argument('--experiment_number', required=True, type=int, help="Number of the experiment")
    parser.add_argument('--aws_prefix', required=True, help="AWS prefix for the table name")
    parser.add_argument('--save_dir', default='./results', help="Directory to save the combined dataset")
    parser.add_argument('--get_partial_trials', action='store_true', help="Fetch partial trial data if full session data is not available")
    parser.add_argument('--get_all_partial_trials', action='store_true', help="Fetch ALL partial trial data")
    parser.add_argument('--check_if_was_screened_out', action='store_true', help="Check if the participant was screened out (only applicable for experiments with in-task screenouts)")
    parser.add_argument('--require_min_test_trials', type=int, help="If getting partial trials, require this many trials in the 'test' blocks (blocks 8 and 9 by default)")
    parser.add_argument('--test_blocks', type=int, nargs='+', default=[8,9], help="List of test block indices (if using require_min_test_trials option)")
    args = parser.parse_args()

    main(args.experiment_name, args.experiment_number, args.aws_prefix, args.save_dir, args.get_partial_trials, args.get_all_partial_trials, args.require_min_test_trials, args.test_blocks, args.check_if_was_screened_out)