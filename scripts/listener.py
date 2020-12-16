#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64
import os

# HARDWARE = "GPU"
HARDWARE = "CPU"

PARTICLE = 80
cnt = 0

def callback(data):
    global cnt
    cnt += 1
    print("%03d: %s" % (cnt,data.data))
    f.write(str(data.data) + "\r\n")

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("fps", Float64, callback)
    rospy.spin()

if __name__ == '__main__':
    path = __file__.split("/")[:-2]
    path = os.path.join(*path)
    path = "/" + path
    path += "/tools/" + HARDWARE + "/" + str(PARTICLE)
    f = open(path, 'w')
    print("listener")
    print(path)
    listener()
