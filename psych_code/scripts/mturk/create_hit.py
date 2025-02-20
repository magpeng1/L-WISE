import boto3
import argparse
from lwise_psych_modules import load_configs
import pprint


def create_external_question_xml(external_url, frame_height=600):
    """
    Manually generates an XML string for an External Question for MTurk HIT.
    """

    xml_str = f'''<?xml version="1.0" encoding="UTF-8"?>
<ExternalQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd">
    <ExternalURL>{external_url}</ExternalURL>
    <FrameHeight>{frame_height}</FrameHeight>
</ExternalQuestion>'''

    return xml_str


def main():
  parser = argparse.ArgumentParser(description='Create MTurk HIT')
  parser.add_argument('--task_url', type=str, required=True, help='URL of task (from deploy_experiment.py)')
  parser.add_argument('--prod', default=False, action='store_true', help='add --prod to deploy to production. Otherwise, it goes to sandbox')
  parser.add_argument('--config_path', type=str, required=True, help='Path to config .yaml file')
  parser.add_argument('--require_quals', nargs='+', help='One or more qualification type IDs to require. Specify multiple quals like --require_quals AF1SF ADF87A. Masters is required by default.')
  parser.add_argument('--require_quals_to_view', nargs='+', help='One or more qualification type IDs to require for even viewing the HIT. Specify multiple quals like --require_quals_to_view AF1SF ADF87A. ')
  parser.add_argument('--hide_from_quals', nargs='+', help='One or more qualification type IDs to hide the HIT from. Specify multiple quals like --hide_from_quals AF1SF ADF87A')
  args = parser.parse_args()

  args = parser.parse_args()

  # MTurk Sandbox or Production URL
  if args.prod:
    endpoint_url = "https://mturk-requester.us-east-1.amazonaws.com"
  else:
    endpoint_url = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"

  # Create a boto3 client for MTurk
  mturk_client = boto3.client('mturk', endpoint_url=endpoint_url, region_name='us-east-1')

  _, _, hit_config = load_configs(args.config_path)

  # Generate External Question XML
  external_question_xml = create_external_question_xml(args.task_url, int(hit_config["frame_height"]))

  if args.prod:
    print("Requiring masters qualification")
    qualification_reqs = [{
      'QualificationTypeId': '2F1QJWKUDD8XADTFD2Q0G6UTO95ALH',  # Masters ID. See https://docs.aws.amazon.com/AWSMechTurk/latest/AWSMturkAPI/ApiReference_QualificationRequirementDataStructureArticle.html#MasterQualifications
      'Comparator': 'Exists',
      'RequiredToPreview': True  # This ensures only Masters can preview and accept the HIT
    }]
  else:
    qualification_reqs = []

  # Include only workers with specified qualifications
  if args.require_quals:
    for qual_id in args.require_quals:
      qualification_reqs.append({
        'QualificationTypeId': qual_id,
        'Comparator': 'Exists',
        'RequiredToPreview': True  # Workers must have this qualification to preview and accept the HIT
      })

  # Exclude workers with specified qualifications
  if args.hide_from_quals:
    for qual_id in args.hide_from_quals:
      qualification_reqs.append({
        'QualificationTypeId': qual_id,
        'Comparator': 'DoesNotExist',
        'RequiredToPreview': True,  # Workers with this qualification cannot preview or accept the HIT
        'ActionsGuarded': 'DiscoverPreviewAndAccept'
      })

  if "autoapprove" in hit_config and hit_config["autoapprove"]:
    autoapproval_delay = 30 # Seconds
  else:
    autoapproval_delay = 864000 # 10 days = 864000 seconds 


  response = mturk_client.create_hit(
      Title=hit_config["title"],
      Description=hit_config["description"],
      Reward=str(hit_config["reward"]),
      MaxAssignments=int(hit_config["num_respondents"]),
      AssignmentDurationInSeconds=int(hit_config["max_time_seconds"]),
      LifetimeInSeconds=int(hit_config["lifetime_seconds"]),
      Question=external_question_xml,
      QualificationRequirements=qualification_reqs,
      AutoApprovalDelayInSeconds=autoapproval_delay
  )

  print("HIT has been created. HIT ID:", response['HIT']['HITId'])

  if args.prod:
    print("Printing full response...")
    pprint.PrettyPrinter(indent=2).pprint(response)

if __name__ == "__main__":
  main()