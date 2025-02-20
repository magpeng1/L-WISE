import os
import sys
import boto3
import numpy as np
import xarray as xr
import argparse
import pprint
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) # Add parent directory to pythonpath
from get_mturk_assignment import get_mturk_assignment  # Make sure this script is in the same directory
from mturk_qual import *
from botocore.exceptions import ClientError

def fetch_assignment_ids(hit_id, sandbox, get_exp_id=True):
    """Fetch all assignment IDs, and the exp_id for a given HIT ID."""
    endpoint_url = "https://mturk-requester-sandbox.us-east-1.amazonaws.com" if sandbox else "https://mturk-requester.us-east-1.amazonaws.com"
    mturk_client = boto3.client('mturk', endpoint_url=endpoint_url, region_name='us-east-1')
    assignments = []
    next_token = None

    while True:
        if next_token:
            response = mturk_client.list_assignments_for_hit(HITId=hit_id, NextToken=next_token)
        else:
            response = mturk_client.list_assignments_for_hit(HITId=hit_id)

        assignments.extend(response['Assignments'])
        next_token = response.get('NextToken')
        if not next_token:
            break

    assignment_ids = [assignment['AssignmentId'] for assignment in assignments]

    if get_exp_id:
        response = mturk_client.get_assignment(AssignmentId=assignment_ids[0])
        exp_id = response['HIT']['Question'].split('<ExternalURL>https://')[1].split('</ExternalURL>')[0].split('/')[1].split('.html')[0]
        return assignment_ids, exp_id
    else:
        return assignment_ids

def combine_datasets(datasets):
    """Combine multiple xarray datasets into one with a new dimension for participants."""
    combined = xr.concat(datasets, dim='participant')
    return combined

def fetch_assignment_metadata(dynamodb_client, assignment_ids, exp_id, aws_prefix="morgan"):
    """Fetch additional metadata from DynamoDB for each participant using scan."""
    table_name = aws_prefix + "_" + exp_id + "_trialset_id_mapper"
    metadata = []

    for assignment_id in assignment_ids:
        response = dynamodb_client.scan(
            TableName=table_name,
            FilterExpression="assignment_id = :assignment_id",
            ExpressionAttributeValues={":assignment_id": {"S": assignment_id}}
        )
        items = response['Items']
        # Assuming there's only one entry per assignment_id, so we take the first result
        if items:
            metadata.append(items[0])
        else:
            metadata.append(None)

    assignments = {d["assignment_id"]["S"]:{
        "hit_id": d["hit_id"]["S"],
        "worker_id": d["worker_id"]["S"],
        "trialset_id":  d["trialset_id"]["N"],
    } for d in metadata}

    return assignments

def main(hit_id, save_dir, sandbox=False, approve_all_assignments=False, pay_bonuses=False, assign_qualification=False, qualification_name=None, qualification_type_id=None, save=True):
    # NOTE about qualifications: 
    # If you set assign_qualification to True, you must provide a qualification_name. Suggest using the name of the experiment, like shorebirds_v3_learn_0. 
    # If you already have a qualification, use qualification_type_id - otherwise, leave this as None. 
    assignment_ids, exp_id = fetch_assignment_ids(hit_id, sandbox, get_exp_id=True)

    try:
        # Initialize a DynamoDB client
        dynamodb_client = boto3.client('dynamodb', region_name='us-east-1')
        assignments = fetch_assignment_metadata(dynamodb_client, assignment_ids, exp_id)

        print("Assignment metadata:")
        pprint.PrettyPrinter(indent=2).pprint(assignments)
    except:
        print("Could not fetch assignment metadata. This could pose a problem for approving assignments and paying bonuses.")

    if sandbox:
        endpoint_url = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"
    else:
        endpoint_url = "https://mturk-requester.us-east-1.amazonaws.com"

    # Create a boto3 client for MTurk
    mturk_client = boto3.client('mturk', endpoint_url=endpoint_url, region_name='us-east-1')

    if assign_qualification:
        assert qualification_type_id is not None or qualification_name is not None, "If creating a new qualification, must provide a qualification name (perhaps the name of the experiment, like shorebirds_v3_learn_0)."
    
        if qualification_type_id is None: 
            qualification_type_id = create_qualification(mturk_client, qualification_name)


    datasets = []
    total_bonuses = 0
    for assignment_id in assignment_ids:
        print("----------------")
        save_path = os.path.join(save_dir, f"{assignment_id}.nc")
        ds, bonus = get_mturk_assignment(assignment_id, save_path, sandbox, verbose=False, save=False)

        for mixed_type_var in ['perf', 'reaction_time_msec']:
            if mixed_type_var in ds:
                ds[mixed_type_var] = ds[mixed_type_var].astype(np.float64)

        if ds is not None:
            datasets.append(ds)

        if approve_all_assignments:
            print("Approving assignment...")
            if ds is None:
                print("WARNING: data from this assignment was not processed successfully. Do you want to continue? (type y or n):")
                yes = {'yes','y', 'ye'}
                no = {'no','n'}
                while True:
                    choice = input().lower()
                    if choice in yes:
                        break
                    elif choice in no:
                        print("Exiting...")
                        exit()
                    else:
                        sys.stdout.write("Please respond with 'yes' or 'no'")
                feedback = "We are sorry that a technical problem prevented your full participation. We will fix this issue for future HITs. For this time, we are giving the full original amount of compensation."
            else:
                feedback=""
            
            try:
                mturk_client.approve_assignment(
                    AssignmentId=assignment_id,
                    RequesterFeedback=feedback
                )
                print("Approved.")
            except ClientError as error:
                if error.response['Error']['Code'] == 'RequestError':
                    error_message = error.response['Error']['Message']
                    if 'status of: Submitted' in error_message:
                        print(f"Assignment {assignment_id} already approved.")
                    else:
                        raise error
                else:
                    raise error

        if bonus is not None and pay_bonuses:
            try:
                bonus_str = f"{bonus:.2f}"
                print(f"Paying bonus of ${bonus_str}...")
                mturk_client.send_bonus(
                    WorkerId=assignments[assignment_id]["worker_id"],
                    BonusAmount=bonus_str,
                    AssignmentId=assignment_id,
                    Reason="Bonus for accurate responses",
                    UniqueRequestToken=assignment_id
                )
                total_bonuses = total_bonuses + bonus
                print("Bonus payment successful")
            except ClientError:
                print("Bonus payment failed. This participant may have already received a bonus.")

        if assign_qualification:
            assign_custom_qualification(qualification_type_id=qualification_type_id, worker_id=assignments[assignment_id]["worker_id"], mturk=mturk_client)

    combined_ds = combine_datasets(datasets)

    if pay_bonuses:
        print(f"Total bonuses delivered: {total_bonuses}")

    # Save the combined dataset
    if save:
        combined_save_path = os.path.join(save_dir, f"combined_dataset_{hit_id}.h5")
        if os.path.isfile(combined_save_path):
            print("Overwriting existing version of", combined_save_path)
            os.remove(combined_save_path)
        print(f"Saving to: {combined_save_path}")

        combined_ds['reaction_time_msec'] = combined_ds['reaction_time_msec'].astype(float)

        combined_ds.to_netcdf(path=combined_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine MTurk assignment data into a single dataset")
    parser.add_argument('--hit_id', required=True, help="HIT ID to fetch assignments for")
    parser.add_argument('--save_dir', default='./results', help="Directory to save datasets")
    parser.add_argument('--no_save', action='store_true', help="Do not save (e.g., if just approving HITs)")
    parser.add_argument('--sandbox', action='store_true', help="Use MTurk sandbox")
    parser.add_argument('--approve_all_assignments', action='store_true', help="Automatically approve ALL assignments")
    parser.add_argument('--pay_bonuses', action='store_true', help='pay all calculated bonuses.')
    parser.add_argument('--assign_qualification', action='store_true', help='Assign all participants a qualification (must also specify either qualification_name for a new qual or qualification_type_id for an existing one.)')
    parser.add_argument('--qualification_name', default=None, help="Name of NEW qualification to create and assign - e.g. shorebirds_v3_learn_0")
    parser.add_argument('--qualification_type_id', default=None, help="ID of an existing qualification to assign to all participants")
    args = parser.parse_args()

    main(args.hit_id, args.save_dir, args.sandbox, approve_all_assignments=args.approve_all_assignments, pay_bonuses=args.pay_bonuses, assign_qualification=args.assign_qualification, qualification_name=args.qualification_name, qualification_type_id=args.qualification_type_id, save=(not args.no_save))
