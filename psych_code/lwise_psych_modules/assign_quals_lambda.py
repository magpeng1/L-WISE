import boto3
import json
import os
import urllib3

MTURK_CLIENT = boto3.client('mturk', endpoint_url="https://mturk-requester.us-east-1.amazonaws.com")
HTTP = urllib3.PoolManager()

def lambda_handler(event, context):
    if 'body' in event:
        body = event['body'] if isinstance(event['body'], dict) else json.loads(event['body'])
    else:
        body = event

    worker_id = body['worker_id']
    qualification_type_ids = body['qualification_type_ids']
    if isinstance(qualification_type_ids, str): # If just one ID provided, put it in a list of length 1
        qualification_type_ids = [qualification_type_ids]

    if 'platform' in body and body['platform'] == 'prolific':
        headers = {
            "Authorization": f"Token {os.environ.get('PROLIFIC_API_TOKEN')}",
            "Content-Type": "application/json"
        }
        
        for group_id in qualification_type_ids:
            url = f"https://api.prolific.com/api/v1/participant-groups/{group_id}/participants/"
            prolific_body = json.dumps({"participant_ids": [worker_id]}).encode('utf-8')
            response = HTTP.request('POST', url, body=prolific_body, headers=headers)
            if response.status == 200:
                print(f"Successfully added participant to group {group_id}")
            else:
                print(f"Failed to add participant to group {group_id}. Status code: {response.status}")
                print(f"Request body: {prolific_body}")
                print(f"Response: {response.data.decode('utf-8')}")
                return {
                    'statusCode': response.status,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'  # This should match the CORS settings in the API gateway. But, if the API gateway is stricter, then its rules will be applied
                    },
                    'body': json.dumps({
                        'message': f"Failed to add participant to group {group_id}",
                        'response': response.data.decode('utf-8')
                    })
                }
    else: # Assume mturk
        
        sandbox = body['sandbox']
        
        if 'score' in body: 
            score = body['score']
        else:
            score = -1

        if sandbox:
            mturk = boto3.client('mturk', endpoint_url="https://mturk-requester-sandbox.us-east-1.amazonaws.com")
        else:
            mturk = MTURK_CLIENT

        try:
            for qualification_type_id in qualification_type_ids:
                response = mturk.associate_qualification_with_worker(
                    QualificationTypeId=qualification_type_id,
                    WorkerId=worker_id,
                    IntegerValue=score,
                    SendNotification=False
                )
                print(f"Qualification {qualification_type_id} assigned to worker {worker_id}")

        except Exception as e:
            print(f"Error assigning qualifications: {str(e)}")
            raise e

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'  # This should match the CORS settings in the API gateway. But, if the API gateway is stricter, then its rules will be applied
        },
        'body': json.dumps('Qualifications/group membership assigned successfully')
    }