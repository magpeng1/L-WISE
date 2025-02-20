from lwise_psych_modules import *
import argparse
            
def main(bucket_names):
    for name in bucket_names:
        print(f"Emptying S3 bucket '{name}'...")
        empty_s3_bucket(name)
        print(f"Deleting S3 bucket '{name}'...")
        delete_s3_bucket(name)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and combine experiment data from DynamoDB and S3")
    parser.add_argument('--bucket_names', type=str, nargs='+', required=True, help="Names of buckets to delete. E.g. --bucket_names bucket1 bucket2")
    args = parser.parse_args()

    main(args.bucket_names)