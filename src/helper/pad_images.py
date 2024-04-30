import argparse
from pathlib import Path
import cv2
from glob import glob
import shutil

border_type = cv2.BORDER_CONSTANT

def main(opt):
  img_output_folder = opt.output_folder / "images"
  img_exts = [".jpg", ".jpeg", ".png"]
  for img_ext in img_exts:
    for img_filepath in glob(str(opt.images_folder) + "/*" + img_ext):
      img = cv2.imread(img_filepath)
      height, width, _ = img.shape
      right = max(0, opt.img_width - width)
      bottom = max(0, opt.img_height - height)
      new_img = cv2.copyMakeBorder(img, 0, bottom, 0, right, border_type, value=[0, 0, 0])

      img_output_filepath = img_output_folder / Path(img_filepath).name
      cv2.imwrite(str(img_output_filepath), new_img)

      label_filepath = opt.labels_folder / (str(Path(img_filepath).stem) + ".txt")
      with open(label_filepath, 'r') as f:
        annotations = f.readlines()
      
      new_annotations = []
      for annotation in annotations:
        # class, xc, yc, width, height (normalised)
        cat, xc_norm, yc_norm, w_norm, h_norm = annotation.split(" ")
        new_xc_norm = (float(xc_norm) * width) / max(width, opt.img_width)
        new_yc_norm = (float(yc_norm) * height) / max(height, opt.img_height)
        new_w_norm = (float(w_norm) * width) / max(width, opt.img_width)
        new_h_norm = (float(h_norm) * height) / max(height, opt.img_height)
        new_annotations.append([cat, str(new_xc_norm), str(new_yc_norm), str(new_w_norm), str(new_h_norm)])

      label_output_filepath = opt.output_folder / "labels" / label_filepath.name
      with open(str(label_output_filepath), 'w') as f:
        for annotation in new_annotations:
          # write each item on a new line
          f.write(" ".join(annotation) + "\n")
  print("Done!")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--images-folder', type=Path, required=True, help='target number of images to generate')
  parser.add_argument('--labels-folder', type=Path, required=True, help='target number of images to generate')
  parser.add_argument('--img-width', type=int, required=True, help='target img width')
  parser.add_argument('--img-height', type=int, required=True, help='target img height')
  parser.add_argument('--output-folder', type=Path, required=True, help='output folder for labels and images')
  opt = parser.parse_args()
  print(opt)

  main(opt)