import os
import boto3
import requests

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


digits_string = list(map(str, range(10))) # [str(i) for i in range(10)]

def is_ip(ip):
    return all([char == '.' or char in digits_string for char in ip])

class EC2Instance:    
    def __init__(self, update_ip=True, reboot=False, test=False, instance_id='i-01796a60ab18b8bd5'):
        self.ip = None
        self.url = None
        self.instance_id = instance_id
        if update_ip:
            self.update_ip()
        if reboot:
            self.reboot()
        if test:
            self.test()

    def update_ip(self):
        ip = get_public_ipv4(self.instance_id)
        self.url = ""
        if 'ip' in ip:
            self.ip = ip["ip"]
            self.url = f"http://{self.ip}"
            return self.url
        print("ERROR IN GET EC2 IP: ", ip["error"])

    def reboot(self):
        reboot_ec2_instance(self.instance_id)

    def test(self):
        try:
            if not self.url:
                raise Exception("AWS EC2 INSTANCE URL NOT FOUND")
            res = requests.get(f'{self.url}/init')
            instance_state = res.ok
        except Exception as e:
            instance_state = False
            print("EC2 INSTANCE TEST REQUEST FAILED. ERROR:", str(e))
        print("EC2 INSTANCE OK:", instance_state)
        return instance_state
    
    
# Example Usage
    
# # Replace this with the ID of the EC2 instance you want to reboot
# instance_id = 'i-01796a60ab18b8bd5'

# # Call the reboot_ec2_instance function with the instance_id
# reboot_ec2_instance(instance_id)