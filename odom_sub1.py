#! usr/bin/env python3

from calendar import c
from cmath import pi
import rospy 
import carla
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
import numpy as np 
import cv2 
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import BoundingBoxes
from carla_msgs.msg import CarlaActorList, CarlaActorInfo
import time 

client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()
carla_map = world.get_map()

bridge = CvBridge()

rospy.init_node('carla_odom_sub')


## This gets the actor information from carla, this needs to be done while the simulator is on 
## or else we cant get this information, we are getting the global position of the ego vehicle camera so that 
## later on we can use this information 

actors = rospy.wait_for_message('carla/actor_list', CarlaActorList)

for actor in actors.actors:
    if actor.rolename == 'rgb_front':
        camera = world.get_actor(actor.id)


class Listener:

    global bridge
    global camera
    
    def __init__(self):
        
        self.depth_image = None 

        msg = rospy.wait_for_message('/carla/ego_vehicle/rgb_front/camera_info', CameraInfo)

        self.K = np.array(msg.K).reshape((3,3))

        self.sub = rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, callback=self.darknet_callback)
        self.sub1 = rospy.Subscriber('/carla/ego_vehicle/depth_front/image', Image, callback=self.depth_callback)
        
        # self.sub1 = message_filters.Subscriber('/carla/ego_vehicle/odometry', Odometry, callback=self.callback,queue_size=1)
        # self.sub2 = message_filters.Subscriber('/carla/ego_vehicle/rgb_front/image', Image, callback=self.callback, queue_size=1)
        # ts = message_filters.TimeSynchronizer([self.sub1, self.sub2], 20)
        # ts.registerCallback(self.callback)

    """
    This darknet callback function basically takes in the msg which is the darknet bounding box, and then we can compute the center for the detected object 
    we also get the depth of the image for that particular pixel from another callback and pass it on into this function. The camera to world function basically converts
    the x,y,z from the detection to the global x,y,z 
    """

    def darknet_callback(self,msg1):

        msg_list = msg1.bounding_boxes
        msg_class = msg_list[0].Class
        msg_xmin = msg_list[0].xmin 
        msg_ymin = msg_list[0].ymin 
        msg_xmax = msg_list[0].xmax 
        msg_ymax = msg_list[0].ymax 
        msg_center = (((msg_xmin + msg_xmax)/2), ((msg_ymin + msg_ymax)/2))

        if self.depth_image is not None:
            
            depth_coord = self.depth_image[msg_center[1], msg_center[0]]
            
            vect = np.array([msg_center[0], msg_center[1], 1]).reshape(3,1) * depth_coord

            xyz = self.camera_2_world(camera, vect)

            pub = rospy.Publisher('/detected_global_xyz',PoseStamped)
            goal = PoseStamped()

            goal.header.stamp = rospy.Time.now()
            goal.header.frame_id = "map"
            goal.pose.position.x = xyz[0,0]
            goal.pose.position.y = -xyz[1,0]
            goal.pose.position.z = xyz[2,0]

            goal.pose.orientation.x = 0.0
            goal.pose.orientation.y = 0.0
            goal.pose.orientation.z = 0.0
            goal.pose.orientation.w = 1.0

            pub.publish(goal)

    def depth_callback(self,msg2):

        self.depth_image = bridge.imgmsg_to_cv2(msg2)

    # def callback(self,msg2):
        
    #     global bridge
        
    #     K =  np.array([[400.00000000000006, 0.0, 400.0], [0.0, 400.00000000000006, 300.0], [0.0, 0.0, 1.0]])
    
    #     try:
    #         cv_image = bridge.imgmsg_to_cv2(msg2, 'bgr8')
    #     except CvBridgeError as e:
    #         print(e)

    #     r = rospy.Rate(20)
    #     image_pub = rospy.Publisher('/carla_image_publisher',Image,queue_size=1)
    #     image_pub.publish(bridge.cv2_to_imgmsg(cv_image))
    #     r.sleep()

    """
    
    This is a very important function. In here, we pass the actor information which basically comprises of the pose of the actor, id and the frame 
    from the simulator in the global frame. The transformation matrix basically transforms the image coordinates from OpenCV convention to ROS convention. 
    Which is (x,y,z) to (y,-z,x). We multiply the inverse of the transformation matrix with the ( inverse of the intrinsic matrix @ the vector of the point 
    we are interested to transform and rotate ) in the camera frame. Then we multiply the previous matrix with the transform we get from the carla actor which is a transformation
    matrix. This will give us the pose in the global frame.   
    
    """

    def camera_2_world(self,carla_actor, vector):
        
        transformation_matrix = np.array([[0, 1, 0], 
                                        [0, 0, -1], 
                                        [1, 0, 0] ])

        camera_coords = np.dot( np.linalg.inv(transformation_matrix),np.dot( np.linalg.inv(self.K), vector ) )
        sensor_2_world = np.array(carla_actor.get_transform().get_matrix())

        camera_vector = np.ones((4,1))
        camera_vector[:3, :] = camera_coords
        xyz = np.dot( sensor_2_world, camera_vector)

        return xyz

if __name__ == '__main__':
    sub = Listener()
    rospy.spin()
