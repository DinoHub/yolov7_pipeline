# This code does not do ROS publishing and there's no option to disable display

import argparse
from pathlib import Path
import cv2
import numpy as np
import time

import pyrealsense2 as rs
from yolov7.yolov7 import YOLOv7

def create_output_video_writer(task_type, output_folder, fps, width, height):
  # Create a unique filename based on current datetime and task type
  current_datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
  output_filepath = output_folder / (current_datetime + "_" + task_type + '.avi')
  output_filepath.parent.mkdir(parents=True, exist_ok=True)
  
  # Create a video writer object
  # video_writer = cv2.VideoWriter(str(output_filepath), cv2.VideoWriter_fourcc(*'h264'), float(fps), (width, height))
  video_writer = cv2.VideoWriter(str(output_filepath), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))

  return video_writer

def yuyv_to_bgr(color_frame):
  # Convert YUYV format to BGR format
  h = color_frame.get_height()
  w = color_frame.get_width()
  y = np.frombuffer(color_frame.get_data(), dtype=np.uint8)[0::2].reshape(h, w)
  uv = np.frombuffer(color_frame.get_data(), dtype=np.uint8)[1::2].reshape(h, w)
  yuv = np.zeros((h, w, 2), 'uint8')
  yuv[:,:,0] = y
  yuv[:,:,1] = uv
  bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUYV)

  return bgr

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

  # Configure depth and color streams
  pipeline = rs.pipeline()
  config = rs.config()

  # Get device product line for setting a supporting resolution
  pipeline_wrapper = rs.pipeline_wrapper(pipeline)
  pipeline_profile = config.resolve(pipeline_wrapper)
  device = pipeline_profile.get_device()

  found_rgb = False
  for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
      found_rgb = True
      print(f"using device {s}")
      break
  if not found_rgb:
    print("The demo requires a Depth camera with Color sensor")
    exit(0)

  print(f"Stream configs: {args.width}x{args.height} with {args.fps} fps")
  #config.enable_stream(rs.stream.color, args.width, args.height, rs.format.yuyv, args.fps)
  config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

  if args.inference_folder:
    try:
      inf_video_writer = create_output_video_writer('inference', args.inference_folder, args.fps, args.width, args.height)
    except Exception as e:
      print(f"ERROR: Failed to create video file n the specified inference folder ({args.inference_folder}):\n{e}")
      exit(0)
    
  if args.raw_video_folder:
    try:
      raw_video_writer = create_output_video_writer('raw', args.raw_video_folder, args.fps, args.width, args.height)
    except Exception as e:
      print(f"ERROR: Failed to create video file n the specified raw video folder ({args.raw_video_folder}):\n{e}")
      exit(0)

  # Start streaming
  pipeline.start(config)
  print(f"Starting stream, press q to exit")

  try:
    # To track FPS
    start_time = time.time()
    x = 5 # displays the frame rate every 5 seconds
    counter = 0

    while True:
      # Wait for coherent frames
      frames = pipeline.wait_for_frames()
      color_frame = frames.get_color_frame()
      if not color_frame:
        continue

      # Convert YUYV frame to BGR format
      frame = np.asanyarray(color_frame.get_data())
      # frame = yuyv_to_bgr(color_frame)
      if args.raw_video_folder:
        raw_video_writer.write(frame)

      # Perform object detection using YOLOv7
      dets = yolov7.detect_get_box_in([frame], box_format='ltrb', classes=None)[0]

      show_frame = frame.copy()
      for det in dets:
        ltrb, conf, clsname = det
        l, t, r, b = ltrb
        cv2.rectangle(show_frame, (l, t), (r, b), (255, 255, 0), 1)
        cv2.putText(show_frame, f'{clsname}:{conf:0.2f}', (l, t-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

      if args.inference_folder:
        inf_video_writer.write(show_frame)

      # Calculate and display FPS
      counter += 1
      if (time.time() - start_time) > x:
        print(f"Average FPS over past {x} seconds: {counter / (time.time() - start_time)}")
        counter = 0
        start_time = time.time()

      # Display the annotated frame
      cv2.namedWindow('RealSense YOLOv7', cv2.WINDOW_AUTOSIZE)
      cv2.imshow('RealSense YOLOv7', show_frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.destroyAllWindows()
    inf_video_writer.release()
    raw_video_writer.release()

  finally:
    # Stop streaming
    pipeline.stop()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
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
  parser.add_argument('--width', type=int, default=1280, help='Desired width of video')
  parser.add_argument('--height', type=int, default=720, help='Desired height of video')
  parser.add_argument('--fps', type=int, default=30, help='Desired fps of video')
  args = parser.parse_args()

  main(args)