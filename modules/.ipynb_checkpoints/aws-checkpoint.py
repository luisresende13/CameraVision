import os
import boto3

def get_public_ipv4(instance_id):
    ec2_client = boto3.client('ec2', region_name='sa-east-1')  # Replace 'us-west-1' with your desired region
    response = ec2_client.describe_instances(InstanceIds=[instance_id])
    try:
        reservations = response['Reservations']
        if reservations:
            instances = reservations[0]['Instances']
            if instances:
                instance = instances[0]
                public_ip = instance.get('PublicIpAddress')
                return {'ip': public_ip}
    except Exception as e:
        print(f'Request failed. Error: {str(e)}')
        return {'error': str(e)}

# Usage

# instance_id = 'my-instance-id'
# public_ipv4 = get_public_ipv4(instance_id)
# if public_ipv4:
#     print("Public IPv4 GET request successful:", public_ipv4)
# else:
#     print("Failed to retrieve the public IPv4 address.")
# return {'ip': public_ipv4, 'status': public_ipv4 is None}
