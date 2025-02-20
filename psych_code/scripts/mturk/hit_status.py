import boto3
import argparse
import pprint
from datetime import datetime


def main():
  parser = argparse.ArgumentParser(description='Check status for assignments under a given HIT')
  parser.add_argument('--sandbox', default=False, action='store_true', help='add --sandbox to get data from sandbox mturk. Otherwise, prod.')
  parser.add_argument('--hit_id', required=True, type=str, help='MTurk assignment ID for which to get data')
  parser.add_argument('--next_token', type=str, help="Next token (for pagination - may be returned by an initial run of hit_status.py)")
  parser.add_argument('--bonuses', action='store_true', help='Get info about bonuses specifically')
  parser.add_argument('--expire', default=False, action='store_true', help='Immediately expire the HIT')
  args = parser.parse_args()

  # MTurk Sandbox or Production URL
  if args.sandbox:
    endpoint_url = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"
  else:
    endpoint_url = "https://mturk-requester.us-east-1.amazonaws.com"

  # Create a boto3 client for MTurk
  mturk_client = boto3.client('mturk', endpoint_url=endpoint_url, region_name='us-east-1')

  # Print overall status
  hit = mturk_client.get_hit(HITId=args.hit_id)
  pprint.PrettyPrinter(indent=2).pprint(hit)
  hit_status = hit['HIT']['HITStatus']
  print(f'HIT Status: {hit_status}')

  if args.expire:
    # Update the expiration time for the specified HIT
    response = mturk_client.update_expiration_for_hit(
      HITId=args.hit_id,
      ExpireAt=datetime(2015, 1, 1)
    )

    print("HIT expiration updated:", response)

    response = mturk_client.list_hits(
        MaxResults=100
    )
    print(response)

  max_results=25

  if args.bonuses:
    if args.next_token:
      response = mturk_client.list_bonus_payments(
        HITId=args.hit_id,
        NextToken=args.next_token,
        MaxResults=max_results,
    )
    else:
      response = mturk_client.list_bonus_payments(
        HITId=args.hit_id,
        MaxResults=max_results,
    )
    
    bonuses = []
    for bonus in response['BonusPayments']:
      bonuses.append(bonus["BonusAmount"])
    print("Total bonuses delivered:", round(sum([float(b) for b in bonuses]), 2))

  else:
    statuses = ['Submitted','Approved','Rejected']
    if args.next_token:
      response = mturk_client.list_assignments_for_hit(
        HITId=args.hit_id,
        NextToken=next_token,
        MaxResults=max_results,
        AssignmentStatuses=statuses
    )
    else:
      response = mturk_client.list_assignments_for_hit(
        HITId=args.hit_id,
        MaxResults=max_results,
        AssignmentStatuses=statuses
    )

    # Remove answer for readability
    for assignment in response['Assignments']:
      if 'Answer' in assignment:
          del assignment['Answer']

    if len(response['Assignments']) == 0:
      print("No assignments attached to HIT were completed.")
    else:
      session_times = []
      for assignment in response['Assignments']:
        session_time = assignment["SubmitTime"] - assignment["AcceptTime"]
        print("Session time:", session_time)
        session_times.append(session_time.seconds)

      print("Average session time:", (sum(session_times)/len(session_times))/60, "minutes")

    pprint.PrettyPrinter(indent=2).pprint(response)

  if 'NextToken' in response:
    next_token = response['NextToken']
    print(f"--next_token {next_token}")
  else:
    next_token = None


if __name__ == "__main__":
  main()