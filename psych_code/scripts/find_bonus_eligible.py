import argparse
import re
import boto3
from boto3.dynamodb.conditions import Attr
import pandas as pd
from decimal import Decimal

def query_dynamodb_table(experiment_name, experiment_number, aws_prefix, platform, min_bonus, get_emails):
    dynamodb = boto3.resource('dynamodb')

    # Table name
    mapper_table_name = f"{experiment_name}_{experiment_number}_trialset_id_mapper"
    if aws_prefix:
        mapper_table_name = aws_prefix + "_" + mapper_table_name

    # Get the table
    table = dynamodb.Table(mapper_table_name)

    # Query the table for all items with the specified platform
    response = table.scan(
        FilterExpression=Attr('platform').eq(platform)
    )

    items = response['Items']

    # Handle pagination if there are more items
    while 'LastEvaluatedKey' in response:
        response = table.scan(
            FilterExpression=Attr('platform').eq(platform),
            ExclusiveStartKey=response['LastEvaluatedKey']
        )
        items.extend(response['Items'])

    # Filter items based on bonus_usd
    filtered_items = [
        item for item in items
        if 'bonus_usd' in item and Decimal(item['bonus_usd']) >= Decimal(str(min_bonus))
    ]

    # Convert to pandas DataFrame
    df = pd.DataFrame(filtered_items)

    # Save to CSV
    csv_filename = f"{mapper_table_name}_query_results.csv"
    df.to_csv(csv_filename, index=False)

    print(f"Number of entries: {len(df)}")
    print(f"Results saved to {csv_filename}")

    print("WORKER IDS:")
    for _, row in df.iterrows():
      if str(row['worker_id']) not in ['nan', '6306443583c32f8d065b65f1']:
        print(row['worker_id'])
    
    if get_emails:
        def is_valid_email(email):
            return re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email) is not None
        
        print("EMAILS:")
        emails = []
        for _, row in df.iterrows():
            if 'user_email' in row and row['user_email'] and str(row['user_email']) not in ['594a5d31833f6d0001623f8a@email.prolific.co']:
                email = str(row['user_email']).strip()
                if is_valid_email(email):
                    emails.append(email)
        print(", ".join(emails))
          

def main():
    parser = argparse.ArgumentParser(description='Query DynamoDB table and export results.')
    parser.add_argument('--experiment_number', type=int, required=True, help='Experiment number')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--aws_prefix', type=str, help='Optional AWS prefix (e.g., your name, to avoid confusion with other users)')
    parser.add_argument('--platform', type=str, required=True, help='Platform to filter by')
    parser.add_argument('--min_bonus', type=float, required=True, help='Minimum bonus amount in USD')
    parser.add_argument('--get_emails', default=False, action='store_true', help="Get email addresses")
    args = parser.parse_args()

    query_dynamodb_table(args.experiment_name, args.experiment_number, args.aws_prefix, args.platform, args.min_bonus, args.get_emails)

if __name__ == "__main__":
    main()