import os
import boto3

def get_public_ipv4(instance_id, region_name='sa-east-1'):
    ec2_client = boto3.client('ec2', region_name=region_name)  # Replace 'us-west-1' with your desired region
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

# Example Usage

# instance_id = 'my-instance-id'
# public_ipv4 = get_public_ipv4(instance_id)
# if public_ipv4:
#     print("Public IPv4 GET request successful:", public_ipv4)
# else:
#     print("Failed to retrieve the public IPv4 address.")
# return {'ip': public_ipv4, 'status': public_ipv4 is None}


def reboot_ec2_instance(instance_id, region_name='sa-east-1'):
    try:
        # Create a Boto3 EC2 client
        ec2_client = boto3.client('ec2', region_name=region_name)

        # Reboot the EC2 instance
        response = ec2_client.reboot_instances(InstanceIds=[instance_id])

        # The response doesn't contain detailed information about the instance state after reboot
        print("Rebooting EC2 instance:", instance_id)
        print("Reboot is in progress.")

        # Wait for the instance state to change to running after the reboot
        waiter = ec2_client.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id])

        # Get the instance state after the reboot
        instance = ec2_client.describe_instances(InstanceIds=[instance_id])
        instance_state = instance['Reservations'][0]['Instances'][0]['State']['Name']

        if instance_state == 'running':
            print("Reboot was successful.")
            return 'success'
        else:
            msg = f"Reboot failed. Current instance state: {instance_state}"
            print(msg)
            return msg
        
    except Exception as e:
        msg = f"Error: {e}"
        print(msg)
        return msg

# Example Usage
    
# # Replace this with the ID of the EC2 instance you want to reboot
# instance_id = 'i-01796a60ab18b8bd5'

# # Call the reboot_ec2_instance function with the instance_id
# reboot_ec2_instance(instance_id)