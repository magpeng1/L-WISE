import os
import boto3
import argparse
import sys
from lwise_psych_modules.process_session_data import *


def get_mturk_assignment(assignment_id, save_path, sandbox=False, verbose=False, save=False):
  # MTurk Sandbox or Production URL
  if sandbox:
    endpoint_url = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"
  else:
    endpoint_url = "https://mturk-requester.us-east-1.amazonaws.com"

  # Create a boto3 client for MTurk
  mturk_client = boto3.client('mturk', endpoint_url=endpoint_url, region_name='us-east-1')

  print("Getting data for assignment " + str(assignment_id) + "...")

  response = mturk_client.get_assignment(AssignmentId=assignment_id)

  print("Size of response in bytes:", sys.getsizeof(str(response)))

  ds, bonus = process_nafc_assignment_json(response["Assignment"], verbose=verbose)
  
  if bonus:
    print(f"${bonus:.2f} bonus earned (rounded to nearest cent)")

  if ds is None: 
    print("Failed to process data for assignment", assignment_id)
    print("----------------")

  if ds is not None and save:
    ds.to_netcdf(path=save_path)

  exp_id = response['HIT']['Question'].split('<ExternalURL>https://')[1].split('</ExternalURL>')[0].split('/')[1].split('.html')[0]
  print(f"Experiment ID: {exp_id}")

  # if ds is not None:
  #   bonus = compute_bonus_amount(ds)
  #   print(f"${bonus:.2f} bonus earned (rounded to nearest cent)")
  # else:
  #   bonus = None

  return ds, bonus


def main():
  parser = argparse.ArgumentParser(description='Get data')
  parser.add_argument('--sandbox', default=False, action='store_true', help='add --sandbox to get data from sandbox mturk. Otherwise, prod.')
  parser.add_argument('--assignment_id', required=True, type=str, help='MTurk assignment ID for which to get data')
  parser.add_argument('--verbose', default=False, action='store_true', help='Print out lots of things')
  parser.add_argument('--save_dir', type=str, default='./', help='Location to store xarray dataset object as .cdf')
  args = parser.parse_args()

  get_mturk_assignment(args.assignment_id, os.path.join(args.save_dir, args.assignment_id + ".h5"), args.sandbox, args.verbose, save=True)


if __name__ == "__main__":
  main()