# usage: python tiling.py /path/to/dataset /path/to/output --negative-samples-path /path/to/ns/output --size 640
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
from shapely.geometry import Polygon

class Tiler:
  def __init__(self,
               dataset_directory: str,
               output_folder: str,
               negative_samples_path: str = None,
               slice_size: int = 640,
               no_padding: bool = False):
    """
    Initialize the Tiler class with input parameters.

    :param dataset_directory: Path to the dataset directory. Required.
    :param output_folder: Path to the target output folder. Required.
    :param negative_samples_path: Path to where negative samples should be saved. Default: None.
    :param slice_size: Slice size. Default: 640.
    :param no_padding: Do not pad images, tiles smaller than indicated size will be discarded. Default: False.
    """
    self.dataset_directory = Path(dataset_directory)
    self.output_folder = Path(output_folder)
    self.negative_samples_path = Path(negative_samples_path) if negative_samples_path else None
    self.slice_size = slice_size
    self.no_padding = no_padding

  def create_directory(self, path):
    """
    Create a directory at the specified path, handling exceptions.

    :param path: Path to the directory to create.
    """
    try:
      path.mkdir(parents=True, exist_ok=True)
    except FileNotFoundError:
      print(f"Specified directory {path} not found. Unable to create folder.")
    except PermissionError:
      print(f"Permission denied. Unable to create folder {path}.")

  def calculate_num_slices(self, width, height):
    """
    Calculate the number of vertical and horizontal slices based on the given width and height.

    :param width: Width of the image.
    :param height: Height of the image.
    :return: A tuple containing the number of vertical and horizontal slices.
    """
    num_vertical_slices = (height + self.slice_size - 1) // self.slice_size
    num_horizontal_slices = (width + self.slice_size - 1) // self.slice_size

    if self.no_padding:
      num_vertical_slices -= 1
      num_horizontal_slices -= 1
    
    return num_vertical_slices, num_horizontal_slices

  def create_tile_polygon_and_image(self, img, num_row, num_col, width, height):
    """
    Create a polygon and cropped image for a specific tile within the image.

    :param img: The source image.
    :param num_row: Row index of the tile.
    :param num_col: Column index of the tile.
    :param width: Width of the image.
    :param height: Height of the image.
    :return: A tuple containing the polygon and the cropped image for the tile.
    """
    x1 = num_col * self.slice_size
    y1 = num_row * self.slice_size
    x2 = x1 + self.slice_size
    y2 = y1 + self.slice_size
    crop_pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]) # cropped polygon should always be slice_size x slice_size
    cropped_img = img.crop(crop_pol.bounds)

    # Check if tile requires padding
    if width < x2:
      x2 = width
    if height < y2:
      y2 = height
    pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    return pol, cropped_img

  def save_labels(self, slice_labels, slice_labels_path):
    """
    Save a list of labels to a file with the specified path.

    :param slice_labels: A list of labels to be saved.
    :param slice_labels_path: The path where the labels should be saved.
    """
    if len(slice_labels) > 0:
      slice_df = pd.DataFrame(slice_labels, columns=['class', 'x1', 'y1', 'w', 'h'])
      slice_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')

  def save_negative_samples(self, cropped_img, img_filename, label_filename):
    """
    Save a negative sample (image and label) to the specified path.

    :param cropped_img: The cropped image to be saved.
    :param img_filename: The filename for the image.
    :param label_filename: The filename for the label.
    """
    img_path = self.negative_samples_path / 'images' / img_filename
    label_path = self.negative_samples_path / 'labels' / label_filename
    cropped_img.save(str(img_path))
    with open(str(label_path), 'w') as f:
      pass

  def create_tiles(self, 
                   img_filename: str,
                   img: Image.Image,
                   op_imgs_path: Path,
                   op_labels_path: Path,
                   boxes: list):
    """
    Create tiles from a single image and generate labels for each tile.

    :param img_filename: Filename of the image being processed.
    :param img: The image object.
    :param op_imgs_path: Output path for images.
    :param op_labels_path: Output path for labels.
    :param boxes: List of bounding boxes for the image.
    """
    img_name, img_ext = img_filename.stem, img_filename.suffix
    height, width = img.height, img.width

    num_vertical_slices, num_horizontal_slices = self.calculate_num_slices(width, height)

    # Create tiles and find intersection with bounding boxes for each tile
    for num_row in range(num_vertical_slices):
      for num_col in range(num_horizontal_slices):
        pol, cropped_img = self.create_tile_polygon_and_image(img, num_row, num_col, width, height)

        slice_path = str(op_imgs_path / f'{img_name}_{num_row}_{num_col}{img_ext}')
        slice_labels_path = str(op_labels_path / f'{img_name}_{num_row}_{num_col}.txt')

        no_annotations = True
        slice_labels = []
        for box in boxes:
          if pol.intersects(box[1]):
            # Save image
            if no_annotations:
              cropped_img.save(slice_path)
              no_annotations = False

            # Get the smallest polygon (with sides parallel to the coordinate axes) that contains the intersection
            inter = pol.intersection(box[1])
            new_box = inter.envelope 
            
            # Get coordinates of polygon vertices
            try:
              x, y = new_box.exterior.coords.xy
            except AttributeError:
              print(f"AttributeError in: {img_filename}")
              continue
            
            # Normalize width and height for yolo format
            new_width = (max(x) - min(x)) / self.slice_size
            new_height = (max(y) - min(y)) / self.slice_size
            
            # Get central point for the new bounding box and normalize
            centre = new_box.centroid
            new_x = (centre.x - (num_col * self.slice_size)) / self.slice_size
            new_y = (centre.y - (num_row * self.slice_size)) / self.slice_size

            slice_labels.append([box[0], new_x, new_y, new_width, new_height])
        
        # Save txt with labels for the current tile
        self.save_labels(slice_labels, slice_labels_path)

        # If negative_samples_path is indicated & there are no bounding boxes intersecting the current tile, save this tile to a separate folder
        if self.negative_samples_path and no_annotations:
          self.save_negative_samples(cropped_img, f'{img_name}_{num_row}_{num_col}{img_ext}', f'{img_name}_{num_row}_{num_col}.txt')

  def tile_images(self):
    """
    Tile all images in the dataset_directory and generate labels for the tiles.
    Sets up the folder structure and iterates through images.
    """
    # Create necessary folders
    img_path = self.dataset_directory / "images"
    labels_path = self.dataset_directory / "labels"
    op_imgs_path = self.output_folder / "images"
    op_labels_path = self.output_folder / "labels"
    self.create_directory(op_imgs_path)
    self.create_directory(op_labels_path)
    
    if self.negative_samples_path:
      self.create_directory(self.negative_samples_path / "images")
      self.create_directory(self.negative_samples_path / "labels")

    # Tile all images in a loop
    total_imgs = len(list(img_path.iterdir()))
    for t, img_filename in enumerate(img_path.iterdir()):
      if t % 100 == 0:
        print(f"=== {t} out of {total_imgs}")

      try:
        img = Image.open(img_filename)
      except Image.DecompressionBombError:
        print(f"DecompressionBombError. Skipping image...")
        continue

      try:
        label_filepath = labels_path / f'{img_filename.stem}.txt'
        labels = pd.read_csv(label_filepath, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
        
        # We need to rescale coordinates from 0-1 to real image height and width
        labels[['x1', 'w']] = labels[['x1', 'w']] * img.width
        labels[['y1', 'h']] = labels[['y1', 'h']] * img.height
        
        boxes = []
        # Convert bounding boxes to shapely polygons. We need to invert Y and find polygon vertices from center points
        for row in labels.iterrows():
          x1 = row[1]['x1'] - row[1]['w'] / 2
          y1 = (row[1]['y1']) - row[1]['h'] / 2
          x2 = row[1]['x1'] + row[1]['w'] / 2
          y2 = (row[1]['y1']) + row[1]['h'] / 2

          boxes.append((int(row[1]['class']), Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))
      except FileNotFoundError:
        print(f"WARNING: Could not find label {label_filepath}. Assuming image has no annotations.")
      
      self.create_tiles(img_filename, img, op_imgs_path, op_labels_path, boxes)

  def run(self):
    """
    The main function for the tiling process for a given dataset.

    Folder Structure:
    There are two possible folder structures supported:

    1. Dataset directory contains 'images' and 'labels' subdirectories.
       - dataset_directory
         ├── images
         └── labels

    2. Dataset directory contains multiple subfolders, each with 'images' and 'labels' subdirectories.
       - dataset_directory
         ├── folder_1
         │   └── images
         │   └── labels
         ├── folder_2
         │   └── images
         │   └── labels
         └── ...

    In the first case, all images in 'images' are tiled into the output folder.
    In the second case, each subfolder is processed separately, creating subfolders in the output for each.

    The tiling process will create tiles from images and generate labels for each tile.

    """
    print(f"Slice size: {self.slice_size}")
    output_path = self.output_folder

    # Folder Structure 1
    if (self.dataset_directory / "images").is_dir() and (self.dataset_directory / "labels").is_dir():
      print(f"Tiling '{self.dataset_directory}' into '{output_path}'")
      self.tile_images()

    # Folder Structure 2 (subfolders)
    else:
      sub_folders = [x for x in self.dataset_directory.iterdir() if x.is_dir()]
      for i, sub_folder in enumerate(sub_folders):
        img_subfolder = (sub_folder / "images")
        label_subfolder = (sub_folder / "labels")

        if img_subfolder.is_dir() and label_subfolder.is_dir():
          op_folder = output_path / sub_folder.name

          # Start tiling for subfolder
          print(f"{i + 1} of {len(sub_folders)}: Tiling '{sub_folder}' into '{op_folder}'")
          self.dataset_directory = sub_folder
          self.output_folder = op_folder
          self.tile_images()

        else:
          print(f"Cannot find 'images' or 'labels' folder in {sub_folder}. Skipping tiling this folder...")
    
    print(f"Completed. Saved into {str(output_path)}.")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('dataset_directory', type=Path, help='path to the dataset directory')
  parser.add_argument('output_folder', type=Path, help='path to the target output folder')
  parser.add_argument('--negative-samples-path', type=Path, help='path to where the negative samples should be saved')
  parser.add_argument('--size', type=int, default=640, help='slice size (default: 640)')
  parser.add_argument('--no-padding', action='store_true', help='do not pad images, tiles smaller than indicated size will be discarded')
  args = parser.parse_args()

  tiler = Tiler(args.dataset_directory, args.output_folder, args.negative_samples_path, args.size, args.no_padding)
  tiler.run()
