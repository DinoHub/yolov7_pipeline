import argparse
from pathlib import Path
import cv2
import numpy as np
import time
import os

from yolov7.yolov7 import YOLOv7

import rospy
from subscriber import Subscriber
from publisher import Publisher

terminate = False

def create_output_video_writer(task_type, output_folder, fps, width, height):
  try:
    # Create a unique filename based on current datetime and task type
    current_datetime = time.strftime("%Y%m%d_%Hh%Mm%Ss")
    output_filepath = output_folder / (current_datetime + "_" + task_type + '.avi')
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(str(output_filepath.parent), 0o777)
    
    # Create a video writer object
    # video_writer = cv2.VideoWriter(str(output_filepath), cv2.VideoWriter_fourcc(*'h264'), float(fps), (width, height))
    video_writer = cv2.VideoWriter(str(output_filepath), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))
    
  except Exception as e:
    print(f"ERROR: Failed to create video file in the specified {task_type} videos' folder ({args.output_folder})")
    raise e

  return video_writer

def signal_handler(sig, frame):
    global terminate
    print("\nCtrl+C detected. Stopping program...")
    terminate = True

def main(args):
  # Check if weights file exists
  if not args.weights.is_file():
    raise FileNotFoundError(f"Unable to find weights file: {args.weights}")

  # Check if cfg file exists
  if not args.cfg.is_file():
    raise FileNotFoundError(f"Unable to find cfg file: {args.cfg}")

  # Initialise YOLOv7 inference model
  yolov7 = YOLOv7(
    weights=args.weights,
    cfg=args.cfg,
    bgr=args.bgr,
    gpu_device=args.gpu_device,
    model_image_size=args.model_image_size,
    max_batch_size=args.max_batch_size,
    half=args.half,
    same_size=args.same_size,
    conf_thresh=args.conf_thresh,
    nms_thresh=args.nms_thresh,
    trace=args.trace,
    cudnn_benchmark=args.cudnn_benchmark,
  )

  # Init publisher
  publishers_list = []
  if args.publish_bbox:
    publishers_list.append('bbox')
  if args.publish_depth_params:
    publishers_list.append('depth_frame')
    publishers_list.append('cam_info')
  if args.publish_img_chip:
    publishers_list.append('chip')
  rospy.init_node('cv', anonymous=True)
  pub = Publisher(publishers_list)
  
  # Init subscriber
  sub = Subscriber("/camera/color/image_raw") # Dk put topic here or inside better

  rate = rospy.Rate(args.fps)
  # frame_interval = 1.0 / args.fps

  # To track FPS
  start_time = time.time()
  x = 5 # displays the frame rate every 5 seconds
  counter = 0
  first_successful_frame = True

  print(f"Starting stream, press q to exit if display is on. Otherwise, Ctrl+C.")
  try:
    while not terminate and not rospy.is_shutdown():
      # Wait for coherent frames
      frame_info = sub.get_frame_info()
      frame = frame_info['img']
      if frame is None:
        continue
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      
      if first_successful_frame:
        vid_width = frame_info['width']
        vid_height = frame_info['height']
        if args.inference_folder:
          inf_video_writer = create_output_video_writer('inference', args.inference_folder, args.fps, vid_width, vid_height)
        if args.raw_video_folder:
          raw_video_writer = create_output_video_writer('raw', args.raw_video_folder, args.fps, vid_width, vid_height)
        
        first_successful_frame = False

      # Perform object detection using YOLOv7
      dets = yolov7.detect_get_box_in([frame], box_format='ltrb', classes=None)[0]
      
      # Info for publishing
      frame_id = frame_info['frame_id']
      raw_timestamp = frame_info['timestamp']
      timestamp = f"{raw_timestamp.secs}.{raw_timestamp.nsecs}".replace(".", "")
      
      # Publish if there are detections
      if dets:
        if args.publish_bbox:
          pub.pub_bbox(dets, timestamp, frame_id)
        # TODO depth_params if required
        # if args.publish_depth_params:
        #   pass
        if args.publish_img_chip:
          img_chip = frame[t:b, l:r]
          pub.pub_chip(img_chip, timestamp, frame_id)

      show_frame = frame.copy()
      for det in dets:
        ltrb, conf, clsname = det
        l, t, r, b = ltrb
        cv2.rectangle(show_frame, (l, t), (r, b), (255, 255, 0), 1)
        cv2.putText(show_frame, f'{clsname}:{conf:0.2f}', (l, t-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

      # Write videos
      if args.raw_video_folder: raw_video_writer.write(frame)
      if args.inference_folder: inf_video_writer.write(show_frame)

      # Calculate and display FPS
      counter += 1
      if (time.time() - start_time) > x:
        print(f"Average FPS over past {x} seconds: {counter / (time.time() - start_time)}")
        counter = 0
        start_time = time.time()

      # Display the annotated frame
      if args.display:
        cv2.namedWindow('RealSense YOLOv7', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense YOLOv7', show_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

      rate.sleep()
      # rospy.sleep(frame_interval) # Sleep to maintain the desired frame rate
  
  except KeyboardInterrupt:
    print("KeyBoardInterrupt: Stopping program...")

  finally:
    # Clean up
    if args.display: cv2.destroyAllWindows()
    if args.raw_video_folder: raw_video_writer.release()
    if args.inference_folder: inf_video_writer.release()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # General parameters
  parser.add_argument('--publish-bbox', action='store_true', help='Publish bounding boxes to ROS')
  parser.add_argument('--publish-img-chip', action='store_true', help='Publish image chip to ROS')
  parser.add_argument('--publish-depth-params', action='store_true', help='Publish depth parameters to ROS')
  parser.add_argument('--display', action='store_true', help='Display video with cv2.imshow()')
  
  # YOLOv7 parameters
  parser.add_argument('-w', '--weights', type=Path, required=True, help='Path to YOLOv7 weights')
  parser.add_argument('-c', '--cfg', type=Path, required=True, help='Path to YOLOv7 config file')
  parser.add_argument('--bgr', action='store_true', help='Set to use BGR color space (default: RGB)')
  parser.add_argument('--gpu_device', type=int, default=0, help='GPU device index (default: 0)')
  parser.add_argument('--model_image_size', type=int, default=1280, help='Input image size for the model (default: 640)')
  parser.add_argument('--max_batch_size', type=int, default=64, help='Maximum batch size for inference (default: 64)')
  parser.add_argument('--half', action='store_true', help='Enable half precision inference')
  parser.add_argument('--same_size', action='store_true', help='Use the same size for all images during inference')
  parser.add_argument('--conf_thresh', type=float, default=0.25, help='Confidence threshold for object detection (default: 0.25)')
  parser.add_argument('--nms_thresh', type=float, default=0.45, help='Non-maximum suppression threshold (default: 0.45)')
  parser.add_argument('--trace', action='store_true', help='Enable tracing for better performance')
  parser.add_argument('--cudnn_benchmark', action='store_true', help='Enable cuDNN benchmark mode')
  
  # Video parameters
  parser.add_argument('--inference-folder', type=Path, default=None, help='Folder to save inference videos; videos named by start recording time')
  parser.add_argument('--raw-video-folder', type=Path, default=None, help='Folder to save raw videos (i.e. no annotations); videos named by start recording time')
  parser.add_argument('--fps', type=int, default=30, help='Desired fps of video')
  args = parser.parse_args()

  main(args)
