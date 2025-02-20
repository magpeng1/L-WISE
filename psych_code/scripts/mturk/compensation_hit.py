import argparse
import boto3


def create_custom_qualification_for_worker(worker_id, mturk):

    # Create a unique qualification name based on the worker's ID
    qualification_name = f"comp_{worker_id}"
    qualification_description = f"Custom compensation qualification for worker {worker_id}."

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
  print(f"Custom Qualification assigned to worker {worker_id}")


def create_hit_with_custom_qualification(qualification_type_id, mturk, reward='0.01', assignment_duration=600,
                                        lifetime=86400, max_assignments=1, title="Compensation HIT for specific worker",
                                        description="This is a special HIT created to compensate a specific worker."):
    # Define the HIT layout
    question_xml = """<?xml version="1.0" encoding="UTF-8"?>
<QuestionForm xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2017-11-06/QuestionForm.xsd">
  <Question>
    <QuestionIdentifier>workerid</QuestionIdentifier>
    <DisplayName>Worker ID Verification</DisplayName>
    <IsRequired>true</IsRequired>
    <QuestionContent>
      <Text>Please enter your Worker ID:</Text>
    </QuestionContent>
    <AnswerSpecification>
      <FreeTextAnswer>
        <Constraints>
          <Length minLength="1" maxLength="128"/>
        </Constraints>
        <NumberOfLinesSuggestion>1</NumberOfLinesSuggestion>
      </FreeTextAnswer>
    </AnswerSpecification>
  </Question>
</QuestionForm>
"""

    # Create the HIT with the custom qualification requirement
    response = mturk.create_hit(
        Title=title,
        Description=description,
        Reward=str(reward),
        AssignmentDurationInSeconds=assignment_duration,
        LifetimeInSeconds=lifetime,
        MaxAssignments=max_assignments,
        Question=question_xml,
        QualificationRequirements=[{
            'QualificationTypeId': qualification_type_id,
            'Comparator': 'EqualTo',
            'IntegerValues': [100],  # Assuming 100 is the score assigned to the worker for this qualification
            'RequiredToPreview': True,
        }]
    )

    hit_type_id = response['HIT']['HITTypeId']
    hit_id = response['HIT']['HITId']
    print(f"Created HIT with HITTypeId: {hit_type_id}, HITId: {hit_id}")

    # Return HIT ID for further operations (like retrieving assignment submissions)
    return hit_id


def main(worker_id, prod=True, approve_and_pay_bonus=False, assignment_id=None, bonus_amount=None, feedback=""):

  # Initialize the boto3 MTurk client

  if prod:
      endpoint_url = "https://mturk-requester.us-east-1.amazonaws.com"
  else:
      endpoint_url = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"

  mturk = boto3.client(
      'mturk',
      region_name='us-east-1',  # Update to your region
      endpoint_url=endpoint_url
  )

  if approve_and_pay_bonus:
      assert assignment_id is not None, "If approving and paying bonus, must provide assignment_id"
      assert bonus_amount is not None, "If approving and paying bonus, must provide bonus amount"

      mturk.approve_assignment(
          AssignmentId=assignment_id,
          RequesterFeedback=feedback
      )
      print("Approved assignment", assignment_id)

      print(f"Paying bonus of ${bonus_amount}...")
      mturk.send_bonus(
          WorkerId=worker_id,
          BonusAmount=bonus_amount,
          AssignmentId=assignment_id,
          Reason="Compensation bonus",
          UniqueRequestToken=assignment_id
      )
      print(f"Paid bonus of ${bonus_amount}")

  else:

      # Create qualification
      qualification_type_id = create_custom_qualification_for_worker(worker_id, mturk)
      print(f"Custom qualification created for worker {worker_id}. Qualification Type ID: {qualification_type_id}")

      # Assign qualification to worker
      assign_custom_qualification(qualification_type_id, worker_id, mturk)

      # Create custom HIT for worker
      hit_id = create_hit_with_custom_qualification(qualification_type_id, mturk)
      print(f"HIT created successfully. HIT ID: {hit_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1. Create a custom compensation HIT for a worker, and separately 2. (using --pay_bonus and --assignment_id), pay a bonus")
    parser.add_argument('--worker_id', type=str, required=True, help="Worker ID")
    parser.add_argument('--prod', action='store_true', help="Use MTurk production environment instead of sandbox")
    parser.add_argument('--approve_and_pay_bonus', action='store_true', help="Approve assignment and pay bonus instead of creating compensation HIT.")
    parser.add_argument('--assignment_id', type=str, help="Assignment ID. Required if paying bonus.")
    parser.add_argument('--bonus_amount', type=str, help="Bonus amount as a string. E.g., for $5.53, it should be 5.53")
    parser.add_argument('--feedback', type=str, default="Compensation for a previous HIT.", help="Feedback message for worker")
    args = parser.parse_args()

    main(args.worker_id, prod=args.prod, approve_and_pay_bonus=args.approve_and_pay_bonus, assignment_id=args.assignment_id, bonus_amount=args.bonus_amount, feedback=args.feedback)