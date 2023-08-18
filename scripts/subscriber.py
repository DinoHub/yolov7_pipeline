import rospy
import numpy as np

from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError

class Subscriber():
  """
  A class for subscribing to ROS image topics and extracting image and metadata.

  This class provides methods to initialize the subscriber, handle image callbacks, and extract image and related information.

  Args:
      ros_topic (str): ROS topic name to subscribe to.

  Attributes:
      bridge (CvBridge): A CvBridge object to convert ROS images to OpenCV format.
      _img_sub (rospy.Subscriber): The ROS subscriber object for image messages.
      width (int): Width of the subscribed image.
      height (int): Height of the subscribed image.
      encoding (str): Image encoding format.
      step (int): Number of bytes per row of the image.
      color_img (numpy.ndarray): The color image extracted from the ROS message.
      timestamp (rospy.Time): Timestamp of the received image.
      frame_id (str): Frame identifier of the received image.
  """
  
  def __init__(self, ros_topic):
    """
    Initialize the Subscriber class instance.

    Initializes ROS node, CvBridge, and subscribes to the specified ROS topic.

    Args:
        ros_topic (str): ROS topic name to subscribe to.
    """
    self.width = None
    self.height = None
    self.encoding = None
    self.step = None
    self.color_img = None
    self.timestamp = None
    self.frame_id = None

    # self.bridge = CvBridge()
    self._img_sub = rospy.Subscriber(ros_topic, Image, self._img_callback)

  
  def _img_callback(self, data):
    """
    Callback function for handling incoming image messages.

    Converts the ROS image message to a color image array and extracts relevant information.

    Args:
        data (sensor_msgs.msg.Image): The incoming ROS image message.
    """
    try:
      # color_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
      # self.color_img = color_img
      
      self.width = data.width
      self.height = data.height
      self.encoding = data.encoding
      self.step = data.step

      img_data = np.frombuffer(data.data, dtype=np.uint8)
      self.color_img = img_data.reshape((self.height, self.width, -1))
      self.timestamp = data.header.stamp
      self.frame_id = data.header.frame_id
    except Exception as e:
      print(e)

  def get_frame_info(self):
    """
    Get image and related information from the subscribed message.

    Returns:
        dict: A dictionary containing color_img, width, height, encoding, step, timestamp, and frame_id.
    """
    return {
      'img': self.color_img,
      'width': self.width,
      'height': self.height,
      'encoding': self.encoding,
      'step': self.step,
      'timestamp': self.timestamp,
      'frame_id': self.frame_id
      }

