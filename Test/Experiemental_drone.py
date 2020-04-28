import airsim

import time

from AIgle_Vision.Vision.Camera import Camera

# --> Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.simPause(False)

# client.simEnableWeather(True)
# client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.85)


# --> Setup tools
center_cam = Camera()

while True:
    # --> Reset Drone to starting position
    client.reset()

    # --> Enable API control
    client.enableApiControl(True)
    client.armDisarm(True)

    # state = client.getMultirotorState()
    # s = pprint.pformat(state)
    # print("state: %s" % state)

    client.moveToPositionAsync(0, 0, -2, 3).join()
    client.moveToPositionAsync(0, -40, 0, 16)

    for i in range(2000):
        # cam_tools.display_camera_view(client, "0")
        center_cam.fetch_and_record_single_image(client, file_name=str(i))
        time.sleep(0.01)

    client.hoverAsync().join()

    client.reset()