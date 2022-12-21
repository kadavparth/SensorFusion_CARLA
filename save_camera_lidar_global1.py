#! usr/bin/env python3

import rospy 
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
import numpy as np 
import time 
import csv


class Listener:

    def __init__(self, topic_name, data_class):
        
        self.topic = topic_name
        self.data_class = data_class

        self.sub = rospy.Subscriber(self.topic,self.data_class, callback=self.lidar_callback)
        self.sub = rospy.Subscriber('/hero_pose_xyz',PoseStamped, callback=self.hero_callback)
        self.sub = rospy.Subscriber('/detected_global_xyz',PoseStamped, callback=self.camera_callback)

    def lidar_callback(self,msg):

        rospy.loginfo('Getting Lidar Data')

        x = msg.pose.position.x 
        y = msg.pose.position.y 
        z = msg.pose.position.z 
        t = rospy.Time.now()

        camera_lidar_xyz_20Hz.append([t,self.cam_x,self.cam_y,self.cam_z,x,y,z])
        hero_pose_xyz_20Hz.append([t,self.hero_x,self.hero_y,self.hero_z])

    def hero_callback(self,msg1):

        self.hero_x = msg1.pose.position.x
        self.hero_y = msg1.pose.position.y 
        self.hero_z = msg1.pose.position.z 
    
    def camera_callback(self,msg2):

        self.cam_x = msg2.pose.position.x
        self.cam_y = msg2.pose.position.y 
        self.cam_z = msg2.pose.position.z 

if __name__ == '__main__':
    rospy.init_node('camera_lidar_save_node')
    camera_lidar_xyz_20Hz = []
    hero_pose_xyz_20Hz = []

    topic = '/lidar_xyz_map'
    data_class = PoseStamped

    ls = Listener(topic_name=topic, data_class=data_class)
    rospy.spin()

    file = open('/home/parth/Desktop/uni_stuff/sensor_fusion/project/final_csv_files/cam_lidar_xyz_20Hz.csv','w+', newline='')
    file1 = open('/home/parth/Desktop/uni_stuff/sensor_fusion/project/final_csv_files/hero_xyz_20Hz.csv','w+', newline='')

    with file as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(camera_lidar_xyz_20Hz)
    
    with file1 as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(hero_pose_xyz_20Hz)