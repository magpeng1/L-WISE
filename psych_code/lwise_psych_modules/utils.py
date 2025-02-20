import sys
import yaml
import boto3
from botocore.exceptions import ClientError
import time


class DualOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure the file is updated immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def load_configs(config_path):
  with open(config_path, 'r') as config_file:
    all_configs = yaml.safe_load(config_file)

  trial_config = all_configs.get('trial_config', {})
  session_config = all_configs.get('session_config', {})
  hit_config = all_configs.get('hit_config', {})
  return trial_config, session_config, hit_config


def write_js_vars(js_file_path, var_dict, var_type="let"):
  """
  Make a .js file at location {js_file_path}, in which variables are defined. 
  The .js will have one line of js for each entry in var_dict defining a variable, 
  with the key as the variable name and the value as the new variable's value. 
  IMPORTANT: the value will be entered literally. So if it is supposed to be a string, 
  surround it with quotes within the value itself. 

  Parameters:
  js_file_path : string
    Path to new .js file to be created
  var_dict : dict
    Dict of variables to be defined within the .js
  var_type : string
    Type of variable declaration. var, let, or const.
  """
  with open(js_file_path, 'w') as f:
    for key, value in var_dict.items():
      f.write(f"{var_type} {key} = {value}\n")


def create_s3_bucket(bucket_name, acl="private", allow_public_files=False):
  """Create an S3 bucket and return bucket url."""
  s3_client = boto3.client('s3')
  try:
    s3_client.create_bucket(Bucket=bucket_name, ACL=acl, ObjectOwnership='BucketOwnerPreferred')

    if allow_public_files:
      # Disable block public access settings for the bucket
      s3_client.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
          'BlockPublicAcls': False,
          'IgnorePublicAcls': False,
          'BlockPublicPolicy': False,
          'RestrictPublicBuckets': False
        }
      )

    print(f"Bucket {bucket_name} created successfully with ACL \"{acl}.\" and block_public_access={not allow_public_files}")
  except ClientError as e:
    print(e)
    return False
  return f'https://{bucket_name}.s3.amazonaws.com'


# Check if a bucket exists
def bucket_exists(bucket_name):
    s3_client = boto3.client('s3')
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        return True  # The bucket does exist and you have permission to access it
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            return False  # The bucket does not exist
        elif error_code == '403':
            raise PermissionError("Bucket exists, but you don't have permission to access it. You may need a new bucket name, as bucket names must be globally unique.")
        else:
            raise  # Some other error occurred


def upload_s3_file(bucket_name, file_path, acl="private", object_name=None, loc_in_bucket=None):
  """
  Upload a file to an S3 bucket and make it publicly accessible.

  :param bucket_name: Bucket to upload to
  :param file_path: File to upload
  :param acl: set to "public-read" if you want the file to be publicly readable
  :param object_name: S3 object name. If not specified, file_path is used
  :param loc_in_bucket: Location inside the bucket where the file should be uploaded. E.g. "my_folder/subfolder". If None, uploads at the root.
  :return: True if file was uploaded, else False
  """
  # If S3 object_name was not specified, use the filename
  if object_name is None:
    object_name = file_path.split('/')[-1]

  # If a path inside the bucket is specified, prepend it to the object name
  if loc_in_bucket:
    object_name = f"{loc_in_bucket.rstrip('/')}/{object_name}"

  extra_args = {'ACL': acl}

  # Determine content type based on file extension
  content_type = 'text/html' if file_path.lower().endswith('.html') else None
  if content_type:
    extra_args['ContentType'] = content_type

  # Upload the file
  s3_client = boto3.client('s3')
  try:
    s3_client.upload_file(file_path, bucket_name, object_name, ExtraArgs=extra_args)
    print(f"File {file_path} uploaded to {bucket_name} at {object_name} with ACL {acl}")
    return True
  except ClientError as e:
    print(e)
    return False
  
def empty_s3_bucket(bucket_name):
    s3 = boto3.resource('s3')
    try:
        bucket = s3.Bucket(bucket_name)
        bucket.objects.all().delete()
    except ClientError as e:
        print(f"Error emptying S3 bucket '{bucket_name}': {e}")

def delete_s3_bucket(bucket_name):
    s3 = boto3.client('s3')
    for attempt in range(3):
        try:
            s3.delete_bucket(Bucket=bucket_name)
            print(f"S3 bucket '{bucket_name}' deleted successfully.")
            return
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketNotEmpty':
                print(f"Bucket {bucket_name} is not empty. Retrying in 5 seconds... (Attempt {attempt + 1}/3)")
                time.sleep(5)  # Wait for 5 seconds before retrying
            else:
                print(f"Error deleting S3 bucket '{bucket_name}': {e}")
                return
