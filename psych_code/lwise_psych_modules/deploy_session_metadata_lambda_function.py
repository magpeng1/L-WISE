import argparse
import boto3
from botocore.exceptions import ClientError
import zipfile
import io
import time
from pathlib import Path


def create_lambda_function(lambda_function_name, role_arn, env_vars=None):
    lambda_client = boto3.client('lambda')

    # **** new
    module_dir  = Path(__file__).resolve().parent
    lambda_src  = module_dir / "session_metadata_lambda.py"

    # Zip the Lambda function code
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        zip_file.write(str(lambda_src), "session_metadata_lambda.py") # changed
    zip_buffer.seek(0)

    # Check if the Lambda function already exists
    try:
        lambda_client.get_function(FunctionName=lambda_function_name)
        function_exists = True
    except lambda_client.exceptions.ResourceNotFoundException:
        function_exists = False

    if env_vars is None: 
        env_vars = {}

    # Create the Lambda function
    try:
        if function_exists:
            print(f"WARNING: Lambda function {lambda_function_name} already exists, updating its code...")
            response = lambda_client.update_function_code(
                FunctionName=lambda_function_name,
                ZipFile=zip_buffer.read(),
            )
        else:
            response = lambda_client.create_function(
                FunctionName=lambda_function_name,
                Runtime='python3.8',  # Specify the runtime
                Role=role_arn,  # ARN of the IAM role
                Handler='session_metadata_lambda.lambda_handler',  # Format: <file_name>.<handler_function_name>
                Code={'ZipFile': zip_buffer.read()},
                Description='A Lambda function to generate unique trialset ID and map it to the assignment ID.',
                Timeout=30,  # Execution timeout
                MemorySize=256,  # Memory in MB
                Environment={'Variables': env_vars},
            )

        # Set environment variables if applicable
        if env_vars is not None and len(env_vars) > 0:
            time.sleep(5)
            lambda_client.update_function_configuration(FunctionName=lambda_function_name, Environment={'Variables': env_vars})     

        return response
    except Exception as e:
        print(f"Error creating Lambda function: {e}")
        return None
    

def create_iam_role_for_lambda(lambda_role_name):
    iam = boto3.client('iam')

    # Check if the role already exists
    try:
        roles = iam.list_roles()
        for role in roles['Roles']:
            if role['RoleName'] == lambda_role_name:
                print(f"WARNING: IAM role {lambda_role_name} already exists. Deleting it to replace with a new one...")

                # Detach all managed policies
                attached_policies = iam.list_attached_role_policies(RoleName=lambda_role_name)
                for policy in attached_policies['AttachedPolicies']:
                    iam.detach_role_policy(RoleName=lambda_role_name, PolicyArn=policy['PolicyArn'])
                    print(f"Detached policy {policy['PolicyName']} from role {lambda_role_name}.")

                # Delete all inline policies
                inline_policies = iam.list_role_policies(RoleName=lambda_role_name)
                for policy_name in inline_policies['PolicyNames']:
                    iam.delete_role_policy(RoleName=lambda_role_name, PolicyName=policy_name)
                    print(f"Deleted inline policy {policy_name} from role {lambda_role_name}.")

                # Now, delete the role
                iam.delete_role(RoleName=lambda_role_name)
                print(f"Role {lambda_role_name} has been deleted.")

    except Exception as e:
        print(f"Error checking existing IAM roles: {e}")
        return None

    # Assume Role Policy Document for Lambda
    assume_role_policy_document = '''{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "lambda.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }'''

    try:
        # Create the IAM role
        role_response = iam.create_role(
            RoleName=lambda_role_name,
            AssumeRolePolicyDocument=assume_role_policy_document,
            Description='IAM role for Lambda with DynamoDB access',
        )
        role_arn = role_response['Role']['Arn']
        print(f"IAM Role created: {role_arn}")

        # Attach the basic execution role policy
        basic_execution_policy_arn = 'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
        iam.attach_role_policy(
            RoleName=lambda_role_name,
            PolicyArn=basic_execution_policy_arn
        )
        print(f"Policy {basic_execution_policy_arn} attached to role {lambda_role_name}")

        # Attach the DynamoDB read-write access policy to the role
        policy_arn = 'arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess'
        iam.attach_role_policy(
            RoleName=lambda_role_name,
            PolicyArn=policy_arn
        )
        print(f"Policy {policy_arn} attached to role {lambda_role_name}")

        # Attach the S3 full access policy to the role
        s3_policy_arn = 'arn:aws:iam::aws:policy/AmazonS3FullAccess'
        iam.attach_role_policy(
            RoleName=lambda_role_name,
            PolicyArn=s3_policy_arn
        )
        print(f"Policy {s3_policy_arn} attached to role {lambda_role_name}")

        # IAM roles take a few seconds to propagate for new roles
        time.sleep(10)

        return role_arn
    except Exception as e:
        print(f"Error creating IAM role: {e}")
        return None


def create_api_gateway_api(api_name, lambda_function_name, access_control_allow_origin, delete_old_apis=False, path_part="getTrialSetAndSessionIDs"):
    # Initialize a Boto3 client for API Gateway and Lambda
    apigateway = boto3.client('apigateway')
    lambda_client = boto3.client('lambda')

    # Delete old APIs with the same name
    if delete_old_apis:
        try:
            # Attempt to find an existing API with the specified name
            existing_apis = apigateway.get_rest_apis(limit=500)  # Adjust limit as necessary
            api_id = None
            for i, api in enumerate(existing_apis['items']):
                if api['name'] == api_name:
                    # if i > 0:
                    api_id = api['id']
                    print(f"Found existing API \"{api_name}\" with ID: {api_id}. Deleting...")
                    apigateway.delete_rest_api(restApiId=api_id)
                    print(f"Deleted API \"{api_name}\" with ID: {api_id}.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'TooManyRequestsException':
                print("Unable to delete for now (too many requests). Skipping deletion for the time being and creating new API anyway.")
            else:
                raise e


    # Create a new REST API
    api_response = apigateway.create_rest_api(
        name=api_name,
        description='API for triggering Lambda function to get trialset and session IDs',
        endpointConfiguration={
            'types': ['REGIONAL']
        }
    )
    api_id = api_response['id']
    print(f"API created with ID: {api_id}")

    # Get the root resource ID
    resources = apigateway.get_resources(restApiId=api_id)
    root_id = [resource for resource in resources['items'] if resource['path'] == '/'][0]['id']

    # Create a new resource under the root
    resource_response = apigateway.create_resource(
        restApiId=api_id,
        parentId=root_id,
        pathPart=path_part  # Your resource name
    )
    resource_id = resource_response['id']
    print(f"Resource created with ID: {resource_id}")

    # Create a POST method on the new resource
    method_response = apigateway.put_method(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod='POST',
        authorizationType='NONE'  # Use AWS_IAM or another type if needed
    )
    print("POST method created")

    # Integrate the POST method with the Lambda function
    # uri = f"arn:aws:apigateway:{boto3.session.Session().region_name}:lambda:path/2015-03-31/functions/{lambda_function_name}/invocations"
    uri = f"arn:aws:apigateway:{boto3.session.Session().region_name}:lambda:path/2015-03-31/functions/arn:aws:lambda:{boto3.session.Session().region_name}:{boto3.client('sts').get_caller_identity().get('Account')}:function:{lambda_function_name}/invocations"

    integration_response = apigateway.put_integration(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod='POST',
        type='AWS_PROXY',
        integrationHttpMethod='POST',
        uri=uri
    )
    print("Lambda integration created")

    # Wait for Lambda integration to be set up before proceeding
    time.sleep(5)

    # Create a method response
    apigateway.put_method_response(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod='POST',
        statusCode='200',
        responseModels={
            'application/json': 'Empty'
        },
        responseParameters={
            'method.response.header.Access-Control-Allow-Origin': True,
            'method.response.header.Access-Control-Allow-Headers': True
        }
    )
    print("Method response for POST method configured")

    # Grant API Gateway permission to invoke the Lambda function
    source_arn = f"arn:aws:execute-api:{boto3.session.Session().region_name}:{boto3.client('sts').get_caller_identity().get('Account')}:{api_id}/*/*/*"
    statement_id = 'deploy_session_metadata_lambda_invoke'
    try:
        lambda_client.add_permission(
            FunctionName=lambda_function_name,
            StatementId=statement_id,
            Action='lambda:InvokeFunction',
            Principal='apigateway.amazonaws.com',
            SourceArn=source_arn
        )
        print("Lambda permission granted")
    except lambda_client.exceptions.ResourceConflictException:
        print("Permission already exists. Removing existing permission.")
        lambda_client.remove_permission(
            FunctionName=lambda_function_name,
            StatementId=statement_id
        )
        print("Existing permission removed. Retrying to add permission.")
        
        # Retry adding the permission after removing the existing one
        lambda_client.add_permission(
            FunctionName=lambda_function_name,
            StatementId=statement_id,
            Action='lambda:InvokeFunction',
            Principal='apigateway.amazonaws.com',
            SourceArn=source_arn
        )
        print("Lambda permission granted after removal of existing permission.")
    # Wait for Lambda integration to be set up before proceeding
    time.sleep(5)

    # Enable CORS by setting up the OPTIONS method
    apigateway.put_method(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod='OPTIONS',
        authorizationType='NONE',
        apiKeyRequired=False
    )

    # Set up the method response for the OPTIONS method
    apigateway.put_method_response(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod='OPTIONS',
        statusCode='200',
        responseModels={
            'application/json': 'Empty'
        },
        responseParameters={
            'method.response.header.Access-Control-Allow-Headers': True,
            'method.response.header.Access-Control-Allow-Methods': True,
            'method.response.header.Access-Control-Allow-Origin': True
        }
    )

    # Set up the mock integration for the OPTIONS method
    apigateway.put_integration(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod='OPTIONS',
        type='MOCK',
        requestTemplates={
            'application/json': '{"statusCode": 200}'
        }
    )

    # Set up the integration response for the OPTIONS method with the necessary CORS headers
    apigateway.put_integration_response(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod='OPTIONS',
        statusCode='200',
        responseTemplates={
            'application/json': ''
        },
        responseParameters={
            'method.response.header.Access-Control-Allow-Headers': "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,Accept'",
            'method.response.header.Access-Control-Allow-Methods': "'POST,OPTIONS'",
            'method.response.header.Access-Control-Allow-Origin': f"'{access_control_allow_origin}'"
        }
    )
    print("CORS enabled for OPTIONS method")

    apigateway.put_integration_response(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod='POST',
        statusCode='200',
        responseParameters={
            'method.response.header.Access-Control-Allow-Origin': f"'{access_control_allow_origin}'",
            'method.response.header.Access-Control-Allow-Headers': "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,Accept'"
        }
    )
    print("CORS enabled for POST method")

    # Deploy the API
    deployment_response = apigateway.create_deployment(
        restApiId=api_id,
        stageName='prod'  # Or another stage name
    )
    print(f"API deployed to stage: prod")

    api_endpoint = f"https://{api_id}.execute-api.{boto3.session.Session().region_name}.amazonaws.com/prod/{path_part}"

    return api_endpoint

# Deploy lambda function with associated IAM role and API Gateway endpoint. 
def deploy_session_metadata_lambda_function(lambda_function_name, access_control_allow_origin, delete_old_apis=False, env_vars=None):
    # Define a unique role name for the Lambda execution role
    lambda_role_name = f"{lambda_function_name}_role"

    # Create the IAM role for Lambda
    role_arn = create_iam_role_for_lambda(lambda_role_name)
    if not role_arn:
        print("Failed to create IAM role. Exiting...")
        return

    # Create the Lambda function with the new role
    lambda_response = create_lambda_function(lambda_function_name, role_arn, env_vars)
    if lambda_response:
        print(f"Lambda function created: {lambda_function_name}")

    api_name = lambda_function_name + "_api"
    api_endpoint = create_api_gateway_api(api_name, lambda_function_name, access_control_allow_origin, delete_old_apis=delete_old_apis)
    
    print("API endpoint:", api_endpoint)
    return api_endpoint


def main():
    parser = argparse.ArgumentParser(description='Create API Gateway API to call a lambda function to get trialset ID and session ID, and to store trial/session data')
    parser.add_argument('--session_metadata_lambda_function_name', type=str, required=True, help='Name of lambda function for fetching session metadata (e.g., session ID, trialset ID)')
    parser.add_argument('--access_control_allow_origin', type=str, required=False, default="*", help='The origin of URLs allowed to call this lambda function. E.g. the URL where the experiments main HTML file is stored, like https://my-bucket-name.s3.amazonaws.com')
    parser.add_argument('--delete_old_apis', action='store_true', help='Delete old API gateway APIs to avoid lots of duplicates (mostly for development purposes)')
    args = parser.parse_args()

    deploy_session_metadata_lambda_function(args.session_metadata_lambda_function_name, args.access_control_allow_origin, args.delete_old_apis)
    

if __name__ == "__main__":
    main()
