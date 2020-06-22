# import setup_path
import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()

pose = client.simGetVehiclePose()

while True:
    # teleport the drone + 10 meters in x-direction
    pose.position.z_val -= 10

    client.simSetVehiclePose(pose, True)
