#!/usr/bin/env python3

import rospy 
import cv2 
from std_msgs.msg import String 
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge, CvBridgeError
import numpy as np 
import tf_conversions as tf 

"""
This script is cool, it basically grabs the image from the front camera in CARLA, then we can do 
whatever we want with it, I convert the image from RGB to HSV and then apply a ROI on the image 
then we can get the lane lines with the specified hsv in range values, this works only on white lane lines, 
we need to tweak this for other colors, this is just a proof of concept. 
"""

bridge = CvBridge()

class Listener:

    def __init__(self, topic, data_class):
        
        self.topic = topic 
        self.data_class = data_class
        rospy.init_node('camera_grab_node', anonymous=True)
        sub = rospy.Subscriber(self.topic, self.data_class, self.image_callback)
        rospy.spin()
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print('Shutting down')
        cv2.destroyAllWindows()


    def image_callback(self,msg):
        global bridge

        try:
            cv_image = bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv_image_masked = self.roi_mask_hsv(cv_image)
            cv_image_mask = self.hsv_image(cv_image_masked)
            masked_image = self.roi_mask_hsv(cv_image_mask)
        except CvBridgeError as e:
            print(e)

        cv2.waitKey(1)
        cv2.imshow("Masked", cv_image)
        
        cv2.imshow("Image window", cv_image_mask) # uncomment if you wanna show lane lines in region of interest

    def hsv_image(self,image):

        lowerwhite = (0,0,140)
        upperwhite = (172,111,255)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowerwhite, upperwhite)

        return mask 
    
    def roi_mask_hsv(self,image):
        mask_points = np.array([[0,454], [189,307], [295,307], [511,481]])
        blank_img = np.zeros_like(image)
        image_roi = cv2.fillPoly(blank_img, pts= [mask_points], color=(255,255,255))
        masked = cv2.bitwise_and(image, image_roi)
        return masked


    # def perspective_transform(self, frame):
    #     roi_points = np.array([[0,454], [189,307], [295,307], [511,481]])
    #     desired_roi_points = np.array([[0,0], [0,frame.shape[1]], [frame.shape[0], frame.shape[1]], [frame.shape[0],0]])
    #     # Calculate the transformation matrix
    #     self.transformation_matrix = cv2.getPerspectiveTransform(
    #         roi_points, desired_roi_points)

    #     # Calculate the inverse transformation matrix
    #     self.inv_transformation_matrix = cv2.getPerspectiveTransform(
    #         desired_roi_points, roi_points)

    #     # Perform the transform using the transformation matrix
    #     self.warped_frame = cv2.warpPerspective(
    #         frame, self.transformation_matrix, frame.shape, flags=(
    #             cv2.INTER_LINEAR))

    #     return self.warped_frame

if __name__ == '__main__':

    topic_name = '/carla/ego_vehicle/rgb_front/image'
    data_class_name = Image

    img = Listener(topic_name, data_class_name)
    img.perspective_transform()
