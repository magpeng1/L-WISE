import os
import sys
import boto3
import pandas as pd
from botocore.exceptions import NoCredentialsError
import argparse
from tqdm import tqdm
tqdm.pandas()


def get_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='data/colon_skin_cc', help="Location of dataset")
    parser.add_argument('--csv_path', type=str, default=None, help="Location of csv file indexing the dataset")
    parser.add_argument('--bucket_name', type=str, default='histopath-test-bucket', help="Name of bucket")
    parser.add_argument('--create_new_bucket', default=False, action='store_true')

    return parser.parse_args(argv)


def upload_file_to_s3(s3_client, bucket_name, file_path, s3_file_path):
    try:
        s3_file_path = s3_file_path.replace('\\', '/')  # Ensure the path format is compatible with S3
        s3_client.upload_file(file_path, bucket_name, s3_file_path, ExtraArgs={'ACL': 'public-read'})
        return f'https://{bucket_name}.s3.amazonaws.com/{s3_file_path}'
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except NoCredentialsError:
        print("Credentials not available")
        return None


def main(csv_file_path, bucket_name, data_path, output_json_path, create_new_bucket=False):
    # Initialize S3 client
    s3 = boto3.client('s3')

    # Create S3 bucket
    if create_new_bucket:
        try:
            s3.create_bucket(Bucket=bucket_name, ObjectOwnership='BucketOwnerPreferred')

            # Disable block public access settings for the bucket
            s3.put_public_access_block(
                Bucket=bucket_name,
                PublicAccessBlockConfiguration={
                    'BlockPublicAcls': False,
                    'IgnorePublicAcls': False,
                    'BlockPublicPolicy': False,
                    'RestrictPublicBuckets': False
                }
            )

            # s3.put_bucket_acl(ACL='public-read', Bucket=bucket_name)
        except s3.exceptions.ClientError as e:
            print(f"Error in creating bucket: {e}")
            raise e

    # Load dataframe
    df = pd.read_csv(csv_file_path)

    # Iterate over rows and upload files
    df['url'] = df['im_path'].progress_apply(lambda x: upload_file_to_s3(s3, bucket_name, os.path.join(data_path, x), x))

    # Saving to csv and JSON
    df.to_csv(csv_file_path, index=False)
    df.to_json(output_json_path, orient='records')


if __name__ == "__main__":

    args = get_args(sys.argv[1:])

    if args.csv_path is None:
        csv_path = os.path.join(args.data_path, "dirmap.csv")
    else:
        csv_path = args.csv_path

    main(csv_path, args.bucket_name, args.data_path, os.path.join(args.data_path, "dirmap.json"), args.create_new_bucket)
