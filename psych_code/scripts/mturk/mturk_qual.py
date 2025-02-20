import boto3
import argparse

def create_qualification(mturk, qualification_name, qualification_description=None):
    
    if qualification_description is None: 
      qualification_description = f"Custom qualification for completion of task: {qualification_name}."

    # Create the custom qualification
    create_qual_response = mturk.create_qualification_type(
        Name=qualification_name,
        Description=qualification_description,
        QualificationTypeStatus='Active',
        AutoGranted=False  # Important: set to False so it's not automatically granted to all workers
    )

    qualification_type_id = create_qual_response['QualificationType']['QualificationTypeId']
    print(f"Created custom qualification: {qualification_name} with ID: {qualification_type_id}")

    return qualification_type_id


def assign_custom_qualification(qualification_type_id, worker_id, mturk, integer_value=100):
  response = mturk.associate_qualification_with_worker(
      QualificationTypeId=qualification_type_id,
      WorkerId=worker_id,
      IntegerValue=integer_value,
      SendNotification=False
  )
  print(f"Qualification with id {qualification_type_id} assigned to worker {worker_id}")


def revoke_custom_qualification(qualification_type_id, worker_id, mturk, reason=None):
  if reason is None: 
    response = mturk.disassociate_qualification_from_worker(
      WorkerId=worker_id,
      QualificationTypeId=qualification_type_id,
    )
  else:
    response = mturk.disassociate_qualification_from_worker(
      WorkerId=worker_id,
      QualificationTypeId=qualification_type_id,
      Reason=reason
    )
  print(f"Qualification with id {qualification_type_id} revoked from worker {worker_id}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Combine MTurk assignment data into a single dataset")
  parser.add_argument('--create', action='store_true', help="Create a new qualification")
  parser.add_argument('--assign', action='store_true', help="Assign a qualification (requires --create and --qualification_name or --qualification_type_id)")
  parser.add_argument('--revoke', action='store_true', help="Revoke a qualification (must provide --qualification_type_id)")
  parser.add_argument('--delete', action='store_true', help="Delete a qualification ID")
  parser.add_argument('--sandbox', action='store_true', help="Use MTurk sandbox")
  parser.add_argument('--worker_id', type=str, default=None, help="Mturk worker ID")
  parser.add_argument('--qualification_name', default=None, help="Name of NEW qualification to create - e.g. shorebirds_v3_learn_0")
  parser.add_argument('--qualification_description', default=None, help="Description of new qualification")
  parser.add_argument('--qualification_type_id', default=None, help="ID of an existing qualification to assign")
  parser.add_argument('--integer_value', type=int, default=None, help="Integer value to associate with a qualification being assigned")
  args = parser.parse_args()

  endpoint_url = "https://mturk-requester-sandbox.us-east-1.amazonaws.com" if args.sandbox else "https://mturk-requester.us-east-1.amazonaws.com"
  mturk = boto3.client('mturk', endpoint_url=endpoint_url, region_name='us-east-1')

  qualification_type_id = None

  if args.create:
    assert args.qualification_name is not None, "Must provide a --qualification_name string if creating a new qual"
    assert not args.revoke, "Cannot both assign and revoke a qual simultaneously"
    qualification_type_id = create_qualification(mturk, args.qualification_name)
    print(f"New qualification created: {qualification_type_id}")

  if args.assign:
    assert args.worker_id is not None, "Must provide a worker ID"
    if args.create:
      assert args.qualification_type_id is None, "Conflict: if you are using --create, you must NOT separately specify --qualification_type_id (this is created for you)"
    else:
      assert args.qualification_type_id is not None, "If using --assign, you must provide a --qualification_type_id string."
      qualification_type_id = args.qualification_type_id

    assign_custom_qualification(qualification_type_id, args.worker_id, mturk)

  if args.revoke: 
    assert args.worker_id is not None, "Must provide a worker ID"
    assert not args.create, "Cannot revoke a qual you are creating simultaenously"
    assert args.qualification_type_id is not None, "If using --revoke, you must provide a --qualification_type_id string."
    
    revoke_custom_qualification(args.qualification_type_id, args.worker_id, mturk)

  if args.delete:
    assert args.qualification_type_id is not None, "If using --delete, you must provide a --qualification_type_id string."
    answer = input(f"Are you sure you want to delete qual with id {args.qualification_type_id}? (y/n): ").lower()
    if answer in ['y', 'yes']:
      response = mturk.delete_qualification_type(
          QualificationTypeId=args.qualification_type_id
      )
      print(f"Qualification with id {args.qualification_type_id} deleted")
    else:
      print("Exiting without deleting qual.")

    

