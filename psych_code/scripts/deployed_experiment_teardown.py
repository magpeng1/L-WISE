import argparse
import boto3
from botocore.exceptions import ClientError
from lwise_psych_modules import *

def confirm_teardown(experiment_name, experiment_number):
    print("\nWARNING: This action will permanently delete all resources associated with the experiment.")
    print("To confirm, please re-enter the experiment details:\n")

    confirm_name = input(f"Re-enter the experiment name: ").strip()
    if confirm_name != experiment_name:
        print("Experiment name does not match. Teardown cancelled.")
        return False

    try:
        confirm_number = int(input(f"Re-enter the experiment number: "))
        if confirm_number != experiment_number:
            print("Experiment number does not match. Teardown cancelled.")
            return False
    except ValueError:
        print("Invalid experiment number. Teardown cancelled.")
        return False

    final_confirm = input("\nHave you TRIPLE CHECKED that you want to teardown this experiment and won't lose any important data? This action cannot be reversed. (yes/no): ").lower()
    if final_confirm != 'yes':
        print("Teardown cancelled.")
        return False

    return True

def delete_lambda_function(function_name):
    lambda_client = boto3.client('lambda')
    try:
        lambda_client.delete_function(FunctionName=function_name)
        print(f"Lambda function '{function_name}' deleted successfully.")
    except ClientError as e:
        print(f"Error deleting Lambda function '{function_name}': {e}")

def delete_dynamodb_tables(table_names):
    dynamodb = boto3.client('dynamodb')
    for table_name in table_names:
        try:
            dynamodb.delete_table(TableName=table_name)
            print(f"DynamoDB table '{table_name}' deletion initiated.")
            waiter = dynamodb.get_waiter('table_not_exists')
            waiter.wait(TableName=table_name)
            print(f"DynamoDB table '{table_name}' deleted successfully.")
        except ClientError as e:
            print(f"Error deleting DynamoDB table '{table_name}': {e}")

def delete_api_gateway(api_name):
    apigateway = boto3.client('apigateway')
    try:
        apis = apigateway.get_rest_apis()
        for api in apis['items']:
            if api['name'] == api_name:
                apigateway.delete_rest_api(restApiId=api['id'])
                print(f"API Gateway '{api_name}' deleted successfully.")
                return
        print(f"API Gateway '{api_name}' not found.")
    except ClientError as e:
        print(f"Error deleting API Gateway '{api_name}': {e}")

def delete_iam_role(iam_role_name):
    iam = boto3.client('iam')

    try:
        # Detach all managed policies
        attached_policies = iam.list_attached_role_policies(RoleName=iam_role_name)
        for policy in attached_policies['AttachedPolicies']:
            iam.detach_role_policy(RoleName=iam_role_name, PolicyArn=policy['PolicyArn'])
            print(f"Detached policy {policy['PolicyName']} from role {iam_role_name}.")

        # Delete all inline policies
        inline_policies = iam.list_role_policies(RoleName=iam_role_name)
        for policy_name in inline_policies['PolicyNames']:
            iam.delete_role_policy(RoleName=iam_role_name, PolicyName=policy_name)
            print(f"Deleted inline policy {policy_name} from role {iam_role_name}.")

        # Now, delete the role
        iam.delete_role(RoleName=iam_role_name)
        print(f"Role {iam_role_name} has been deleted.")
    except ClientError as e:
        print(f"Error deleting IAM role': {e}")


def main():
    parser = argparse.ArgumentParser(description='Teardown deployed AWS resources for an experiment.')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--experiment_number', type=int, required=True, help='Experiment number')
    parser.add_argument('--aws_prefix', default='morgan', type=str, help='Prefix for aws resources (e.g., to avoid confusion with other users)')
    parser.add_argument('--force', action='store_true', help="No double-check before teardown. Use at your own risk...")
    args = parser.parse_args()

    if not args.force:
        if not confirm_teardown(args.experiment_name, args.experiment_number):
            print("Teardown cancelled.")
            return

    exp_id = f"{args.experiment_name}_{args.experiment_number}"
    
    # S3 bucket name
    bucket_name = f"{args.experiment_name.replace('_', '-').lower()}-{args.experiment_number}"
    if args.aws_prefix:
        bucket_name = f"{args.aws_prefix}-" + bucket_name

    # Lambda function name
    lambda_function_name = f"{exp_id}_session_metadata"
    if args.aws_prefix:
        lambda_function_name = f"{args.aws_prefix}_" + lambda_function_name

    # API Gateway name
    api_name = lambda_function_name + "_api"

    # IAM role name
    iam_role_name = f"{lambda_function_name}_role"

    # DynamoDB table names
    counter_table_name = f"{args.experiment_name}_{args.experiment_number}_trialset_counter"
    mapper_table_name = f"{args.experiment_name}_{args.experiment_number}_trialset_id_mapper"
    if args.aws_prefix:
        counter_table_name = f"{args.aws_prefix}_" + counter_table_name
        mapper_table_name = f"{args.aws_prefix}_" + mapper_table_name

    # Perform teardown
    print(f"Emptying S3 bucket '{bucket_name}'...")
    empty_s3_bucket(bucket_name)
    
    print(f"Deleting S3 bucket '{bucket_name}'...")
    delete_s3_bucket(bucket_name)
    
    print(f"Deleting Lambda function '{lambda_function_name}'...")
    delete_lambda_function(lambda_function_name)
    
    print(f"Deleting API Gateway '{api_name}'...")
    delete_api_gateway(api_name)

    print(f"Deleting IAM role '{iam_role_name}'...")
    delete_iam_role(iam_role_name)

    print("Deleting DynamoDB tables...")
    delete_dynamodb_tables([counter_table_name, mapper_table_name])

    print("Teardown complete.")

if __name__ == "__main__":
    main()