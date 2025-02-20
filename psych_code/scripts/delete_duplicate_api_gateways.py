import argparse
import boto3
from botocore.exceptions import ClientError
import time

def delete_older_apis(api_name):
    apigateway = boto3.client('apigateway')

    try:
        # Get all REST APIs
        response = apigateway.get_rest_apis(limit=500)  # Adjust limit as necessary
        
        # Filter APIs with the specified name
        matching_apis = [api for api in response['items'] if api['name'] == api_name]
        
        if not matching_apis:
            print(f"No APIs found with the name '{api_name}'.")
            return

        # Sort APIs by creation date, most recent first
        matching_apis.sort(key=lambda x: x['createdDate'], reverse=True)
        
        # Keep the most recent API
        most_recent_api = matching_apis[0]
        print(f"Keeping most recent API: {most_recent_api['name']} (ID: {most_recent_api['id']}, Created: {most_recent_api['createdDate']})")
        
        if len(matching_apis) == 1:
            print('Keeping this one, because there are no other APIs with the same name.')
            return

        # Delete all other matching APIs
        for api in matching_apis[1:]:
            try:
                apigateway.delete_rest_api(restApiId=api['id'])
                print(f"Deleted API: {api['name']} (ID: {api['id']}, Created: {api['createdDate']})")
            except ClientError as e:
                if e.response['Error']['Code'] == 'TooManyRequestsException':
                    print(f"Too many requests error. Waiting 30 seconds before retrying...")
                    time.sleep(30)
                    try:
                        apigateway.delete_rest_api(restApiId=api['id'])
                        print(f"Successfully deleted API after retry: {api['name']} (ID: {api['id']}, Created: {api['createdDate']})")
                    except ClientError as retry_error:
                        print(f"Error deleting API {api['id']} after retry: {retry_error}")
                else:
                    print(f"Error deleting API {api['id']}: {e}")
            
            # Wait 30 seconds before attempting to delete the next API
            print("Waiting 30 seconds before proceeding to the next API...")
            time.sleep(30)

    except ClientError as e:
        print(f"Error accessing API Gateway: {e}")

def main():
    parser = argparse.ArgumentParser(description='Delete older API Gateway APIs with the same name, keeping only the most recent.')
    parser.add_argument('--api_name', type=str, required=True, help='Name of the API Gateway APIs to clean up')
    args = parser.parse_args()

    delete_older_apis(args.api_name)

if __name__ == "__main__":
    main()