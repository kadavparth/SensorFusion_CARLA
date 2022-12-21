#!/usr/bin/env python3

import rospy
import numpy as np
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker
import message_filters
from time import time


"""
This script gets the lidar points from the semantic lidar for the hero vehicle, we then compute the centroid for the points 
and then save them as a vector. We also go from the vehicle frame to the lidar frame and then convert everything to the global frame 
and then publish it to a new topic. 

"""

pub = rospy.Publisher('/detection', Marker, queue_size=10)
pub_map = rospy.Publisher('/detection_map', Marker, queue_size=10)

## This is the position of the lidar from the vehicle frame. the lidar is z meters up from the vehicle frame (translated)

tx, ty, tz = 0.0, 0.0, 2.4

def get_matrix(x, y, z, roll, pitch, yaw):

    """
    This function gets the x,y,z which is the center of the new frame, for example vehicle frame and roll, pitch, yaw
    then returns the rotation + translaion matrix for later on convetring any point to the global frame. 
    
    """
    c_y = np.cos(yaw)
    s_y = np.sin(yaw)
    c_r = np.cos(roll)
    s_r = np.sin(roll)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)

    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix

def create_marker(cx, cy, cz, frame, id):

    """
    This topic creates the marker for the vehicle using the semantic lidar, this then publishes the marker to a topic  
    """

    marker = Marker()

    marker.id = id
    marker.type = 1

    marker.pose.position.x = cx
    marker.pose.position.y = cy
    marker.pose.position.z = cz

    marker.pose.orientation.x = 0.0  
    marker.pose.orientation.y = 0.0  
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    marker.scale.x = 0.5
    marker.scale.y = 0.5
    marker.scale.z = 0.5

    marker.lifetime = rospy.Duration(0.1)

    marker.header.frame_id = frame
    marker.header.stamp = rospy.Time.now()

    if id == 0:
        marker.color.r = 1.0
        marker.color.a = 1.0

        pub.publish( marker )

    else:
        marker.color.g = 1.0
        marker.color.a = 1.0

        pub_map.publish( marker )


def callback(lidar_msg, odom_msg):

    """
    the main callback funtion which reads the points from the lidar message, and the odom message. We compute the 
    center/mean of the x,y,z from the points. Create a marker and publish it to ros to show the center of the detected object. This detected object is 
    in local/laser x,y,z frame. Given the pose of the vehicle and the rigid relation between the lidar and the vehicle we can translate the 
    vehicle frame the lidar frame and then convert everyhting to global frame
    """
    start = time()

    ## read the points for the lidar message which is the semantic lidar   

    gen = pc2.read_points(lidar_msg, skip_nans=True, field_names=("x", "y", "z", "CosAngle", "ObjIdx", "ObjTag"))

    points = []
    
    for idx, p in enumerate(gen):
        
        if p[5] == 10:
            points.append([ p[0], p[1], p[2]] )

    points = np.array(points)

    # centroids for the x,y,z points 
    x = points[:, 0].mean()
    y = points[:, 1].mean()
    z = points[:, 2].mean()

    create_marker( x, y, z, "ego_vehicle/semantic_lidar", id = 0 )

    # vehicle pose from the odom message, this is in the global frame 
    xp = odom_msg.pose.pose.position.x
    yp = odom_msg.pose.pose.position.y
    zp = odom_msg.pose.pose.position.z

    # we can find the r,p,y from the quaternion from the odom message 

    roll, pitch , yaw = euler_from_quaternion([odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
    odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w])

    # this is important function, this moves the vehicle frame to the lidar frame and gets the correct matrix for rotation and translation 
    # to global frame. After multiplying any vector with this matrix, it will convert the point from the lidar frame to the global frame.

    world_matrix = get_matrix(xp + tx, yp + ty, zp + tz, 0, 0, 0)

    # this is the vector for the detetced point from the lidar (the center for the x,y,z for the detetced object)

    vector = np.array([ x, y, z, 1 ]).reshape( (4, 1) )

    # convert from the lidar frame to the global frame for the given vector 

    world_coordinates = world_matrix @ vector

    print(world_coordinates)
    
    # create a marker for the global frame for the detetced objects ceneter. 

    create_marker( world_coordinates[0, 0], world_coordinates[1, 0], \
        world_coordinates[2, 0], "map", id = 1 )

    # print("Elapsed time {} s".format(time() - start))

def main():
    
    rospy.init_node("lidar_segmentation_node")

    rospy.loginfo("Starting lidar node.")

    # rospy.Subscriber('/carla/ego_vehicle/semantic_lidar', PointCloud2, callback)
    cloud_sub = message_filters.Subscriber('/carla/ego_vehicle/semantic_lidar', PointCloud2)
    odom_sub = message_filters.Subscriber('/carla/ego_vehicle/odometry', Odometry)

    ts = message_filters.TimeSynchronizer([cloud_sub, odom_sub], 10)
    ts.registerCallback(callback)

    rospy.spin()

try:
    main()
except KeyboardInterrupt:
    pass