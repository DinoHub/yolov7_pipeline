#!/usr/bin/env python

import numpy as np

import rospy
from vision_msgs.msg import BoundingBox2DArray, BoundingBox2D
from sensor_msgs.msg import Image, CameraInfo

class Publisher():
  """
  A class for publishing various types of messages to ROS topics.

  This class provides methods to initialize publishers and publish messages to specific ROS topics.

  Args:
    msg_formats_list (list): List of message formats to publish.

  Attributes:
    _publishers (dict): Dictionary to store ROS publishers for different message formats.
  """
  
  @classmethod
  def get_ros_topics(cls):
    """
    Get a dictionary of ROS topics and their associated message types.

    Returns:
        dict: A dictionary containing ROS topics as keys and message types as values.
    """
    return {
      'bbox': ('/cv/bounding_box', BoundingBox2DArray),
      'chip': ('/cv/target_chip', Image),
      'depth_frame': ('/cv/depth_frame', Image),
      'cam_info': ('/cv/camera_info', CameraInfo)
    }
  
  def __init__(self, msg_formats_list):
    """
    Initialize the Publisher class instance.

    Initializes ROS node, publishers, and blank messages for the specified message formats.

    Args:
        msg_formats_list (list): List of message formats to publish.
    """
    ros_topics = self.get_ros_topics()
    
    self._publishers = {}
    
    for msg_format in msg_formats_list:
      if msg_format in ros_topics.keys():
        topic, msg_type = ros_topics[msg_format]
        self._init_publisher(msg_format, topic, msg_type)
        self._init_blank_msg(msg_format, msg_type)
      else:
          print(f"ERROR: Invalid msg_format '{msg_format}'. Only the following formats are accepted: {', '.join(ros_topics.keys())}")

  def _init_publisher(self, msg_format, topic, msg_type):
    self._publishers[msg_format] = rospy.Publisher(topic, msg_type, queue_size=1)
    print("init", self._publishers[msg_format], topic)
  
  def _init_blank_msg(self, msg_format, msg_type):
    init_msg = msg_type()
    self._publishers[msg_format].publish(init_msg)

  def _convert_ltrb_to_cwh(self, bbox):
    """
    Convert bounding box coordinates from ltrb to c, w, h where c is (xcentre, ycentre).

    Args:
        bbox (tuple): Bounding box coordinates in ltrb format.

    Returns:
        tuple: A tuple containing c (center), w (width), and h (height) values.
    """
    ltrb, _, _ = bbox
    l, t, r, b = ltrb
    xc = (r - l) / 2.0
    yc = (b - t) / 2.0
    return (xc, yc), r-l, b-t
  
  def pub_bbox(self, dets, timestamp, frame_id):
    """
    Publish bounding box data to the corresponding ROS topic.

    Args:
        dets (list): List of detected bounding boxes.
        timestamp (str): Timestamp for the message.
        frame_id (str): Frame identifier for the message.
    """
    b_msg = BoundingBox2DArray()
    # pose2d_array = []
    
    for i, det in enumerate(dets):
      c, w, h = self._convert_ltrb_to_cwh(det)
      xc, yc = c
      bbox2d = BoundingBox2D()
      bbox2d.center.x = xc
      bbox2d.center.y = yc
      bbox2d.size_x = w
      bbox2d.size_y = h
      b_msg.boxes.append(bbox2d)

    b_msg.header.stamp.secs = int(timestamp[:10])
    b_msg.header.stamp.nsecs = int(timestamp[10:])
    b_msg.header.frame_id = frame_id
    
    self._publishers['bbox'].publish(b_msg)
    print("pub bbox")
  
  def pub_camera_info(self):
    """
    Publish camera information (placeholder method).
    """
    pass
  
  def pub_depth_frame(self, depth_frame, timestamp, frame_id):
    """
    Publish depth frame data to the corresponding ROS topic.

    Args:
        depth_frame (numpy.ndarray): Depth frame data.
        timestamp (str): Timestamp for the message.
        frame_id (str): Frame identifier for the message.
    """
    df_msg = Image()

    df_msg.header.seq = 0
    
    df_msg.header.stamp.secs = int(timestamp[:10])
    df_msg.header.stamp.nsecs = int(timestamp[10:])
    df_msg.header.frame_id = frame_id
    
    h, w, _ = depth_frame.shape
    df_msg.height = h
    df_msg.width = w

    df_msg.encoding = "bgr8"
    df_msg.data = depth_frame.tobytes()
    self._publishers['depth_frame'].publish(df_msg)
    print("pub depth frame")
  
  def pub_chip(self, img_chip, timestamp, frame_id):
    """
    Publish image chip data to the corresponding ROS topic.

    Args:
        img_chip (numpy.ndarray): Image chip data.
        timestamp (str): Timestamp for the message.
        frame_id (str): Frame identifier for the message.
    """
    chip = Image()

    chip.header.seq = 0
    
    chip.header.stamp.secs = int(timestamp[:10])
    chip.header.stamp.nsecs = int(timestamp[10:])
    chip.header.frame_id = frame_id
    
    h, w, _ = img_chip.shape
    chip.height = h
    chip.width = w

    chip.encoding = "bgr8"
    chip.data = img_chip.tobytes()
    self._publishers['chip'].publish(chip)
    print("pub chip")
