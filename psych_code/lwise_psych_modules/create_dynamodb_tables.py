import argparse
import boto3

def create_dynamodb_tables(experiment_name, experiment_number, aws_prefix):
    dynamodb = boto3.resource('dynamodb')

    # Table names
    counter_table_name = f"{experiment_name}_{experiment_number}_trialset_counter"
    mapper_table_name = f"{experiment_name}_{experiment_number}_trialset_id_mapper"
    if aws_prefix: 
        counter_table_name = aws_prefix + "_" + counter_table_name
        mapper_table_name = aws_prefix + "_" + mapper_table_name

    # Create the trialset counter table (if it doesn't exist)
    try:
        counter_table = dynamodb.create_table(
            TableName=counter_table_name,
            KeySchema=[
                {
                    'AttributeName': 'ID',
                    'KeyType': 'HASH'  # Partition key
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'ID',
                    'AttributeType': 'S'
                }
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        counter_table.meta.client.get_waiter('table_exists').wait(TableName=counter_table_name)
        print(f"Table {counter_table_name} created successfully.")

        # Initialize the counter in the table
        counter_table.put_item(
            Item={
                'ID': 'uniqueKey',
                'CurrentCount': -1 # After the first increment, it will be 0, corresponding to the first trialset
            }
        )
        print(f"Initialized CurrentCount in {counter_table_name}.")

    except dynamodb.meta.client.exceptions.ResourceInUseException:
        print(f"WARNING: table {counter_table_name} already exists, leaving it as it is")

    # Create the trialset ID mapper table (if it doesn't exist)
    try: 
        mapper_table = dynamodb.create_table(
            TableName=mapper_table_name,
            KeySchema=[
                {
                    'AttributeName': 'assignment_id',
                    'KeyType': 'HASH'  # Partition key
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'assignment_id',
                    'AttributeType': 'S'
                }
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        mapper_table.meta.client.get_waiter('table_exists').wait(TableName=mapper_table_name)
        print(f"Table {mapper_table_name} created successfully.")
        
    except dynamodb.meta.client.exceptions.ResourceInUseException:
        print(f"WARNING: table {mapper_table_name} already exists, leaving it as it is")

    return counter_table_name, mapper_table_name

def main():
    parser = argparse.ArgumentParser(description='Create DynamoDB tables for an experiment.')
    parser.add_argument('--experiment_number', type=int, required=True, help='Experiment number')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--aws_prefix', type=str, help='Optional AWS prefix (e.g., your name, to avoid confusion with other users)')
    args = parser.parse_args()

    create_dynamodb_tables(args.experiment_name, args.experiment_number, args.aws_prefix)

if __name__ == "__main__":
    main()
