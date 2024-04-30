# Converts datasets in COCO format (images in a folder + coco.json) to YOLO format (only for HBBs)

# Example usage:
# python coco_to_yolo.py /path/to/images /path/to/annotations.json /path/to/output
import argparse
import json
import sys

import cv2
import shutil
from pathlib import Path
from glob import glob

from collections import defaultdict

from verify_coco import COCOVerifier, InvalidCOCOAnnotationError

class ConvertCOCOToYOLO:

    """
    Takes in the path to COCO annotations and outputs YOLO annotations in multiple .txt files.
    COCO annotation are to be JSON formart as follows:
        "annotations":{
            "area":2304645,
            "id":1,
            "image_id":10,
            "category_id":4,
            "bbox":[
                704
                620
                1401
                1645
            ]
        }
        
    """
    class InvalidCOCOAnnotationError(Exception):
        pass

    def __init__(self, img_folder, json_path, output_path, remove_empty, keep_duplicates):
        self.img_folder = img_folder
        self.json_path = json_path
        self.img_output_path = Path(output_path) / "images"
        self.labels_output_path = Path(output_path) / "labels"
        self.remove_empty = remove_empty
        self.keep_duplicates = keep_duplicates
        try:
            shutil.rmtree(str(self.labels_output_path))
            shutil.rmtree(str(self.img_output_path))
        except Exception:
            print("Output folders do not exist, creating folders...")
        else:
            print("Deleted existing contents in output folders...")
        self.img_output_path.mkdir(parents=True, exist_ok=True)
        self.labels_output_path.mkdir(parents=True, exist_ok=True)

    def get_img_shape(self, img_path):
        img = cv2.imread(img_path)
        try:
            return img.shape
        except AttributeError:
            print('AttributeError for ', img_path)
            return (None, None, None)
    
    def normalise_bbox(self, bbox, img_width, img_height):
        x, y, w, h = bbox
        if x < 0 or x + w > img_width or y < 0 or y + h > img_height:
            return None
        if w > img_width or h > img_height:
            return None
        xc = x + (w / 2.0)
        yc = y + (h / 2.0)
        xc /= img_width
        w /= img_width
        yc /= img_height
        h /= img_height
        return (xc, yc, w, h)

    def convert(self,imgs_key='images',annotation_key='annotations',ann_img_id_key='image_id',ann_cat_id_key='category_id',bbox='bbox'):
        # Enter directory to read JSON file
        data = json.load(open(self.json_path))

        if self.keep_duplicates: img_id_to_yolo_annotation = defaultdict(list)
        else: img_id_to_yolo_annotation = defaultdict(set)
        img_id_to_filename = {}
        img_id_to_dimensions = {}
        for img in data[imgs_key]:
          img_id_to_filename[img["id"]] = img["file_name"]
          img_id_to_dimensions[img["id"]] = (img["width"], img["height"])

        # Convert annotations for images with annotations
        for i in range(len(data[annotation_key])):
            image_id = data[annotation_key][i][ann_img_id_key]
            image_filename = img_id_to_filename[image_id]
            category_id = f'{int(data[annotation_key][i][ann_cat_id_key]) - 1}' # yolo category id starts from 0, coco category id starts from 1
            bbox = data[annotation_key][i]['bbox']

            # Convert the data
            yolo_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
            img_width, img_height = img_id_to_dimensions[image_id]
            normalised_yolo_bbox = self.normalise_bbox(yolo_bbox, float(img_width), float(img_height))

            if not normalised_yolo_bbox:
                print(f"WARNING: There was an issue normalising bbox for an annotation in {image_filename}. Annotation details: {data[annotation_key][i]}")
                continue
            
            content =f"{category_id} {normalised_yolo_bbox[0]} {normalised_yolo_bbox[1]} {normalised_yolo_bbox[2]} {normalised_yolo_bbox[3]}"

            if self.keep_duplicates: img_id_to_yolo_annotation[image_id].append(content)
            else: img_id_to_yolo_annotation[image_id].add(content)
        
        # Write annotation files for images with annotations
        for image_id, content in img_id_to_yolo_annotation.items():
            image_filename = img_id_to_filename[image_id]
            filename = f"{Path(image_filename).stem}.txt"
            filepath = Path(self.labels_output_path) / filename
            
            with open(filepath, 'w') as output_file:
                output_file.write("\n".join(content) + "\n")
        
        # Write annotation files for images with NO annotations
        if not self.remove_empty:
            for img in data[imgs_key]:
                image_id = img["id"]
                if image_id not in img_id_to_yolo_annotation:
                    image_filename = img_id_to_filename[image_id]
                    filename = f"{Path(image_filename).stem}.txt"
                    filepath = Path(self.labels_output_path) / filename
                    
                    # Write empty file
                    with open(filepath, 'w'):
                        pass

    def copy_images(self):
        data = json.load(open(self.json_path))

        img_filenames = [Path(image['file_name']).stem for image in data['images']]

        img_exts = [".jpg", ".jpeg", ".png"]
        img_label_files = glob(str(self.labels_output_path) + "/*.txt")
        img_labels = [Path(label_file).stem for label_file in img_label_files]

        total_labels = len(img_labels)
        copied_count = 0

        for idx, img_label in enumerate(img_labels):
            for ext in img_exts:
                img_filepath = Path(self.img_folder) / (img_label + ext)
                op_filepath = self.img_output_path / Path(img_filepath).name
                if img_filepath.exists():
                    if img_filepath.stem in img_filenames:
                        shutil.copy(img_filepath, str(op_filepath))
                        copied_count += 1
                        break
                    else:
                        print(f"WARNING: Image {img_filepath} exists but it does not exist in COCO Json file 'images'. Should not be possible!")
            else:
                print(f"\nWARNING: Unable to find corresponding image for label {img_label}")

            if (idx + 1) % 10 == 0:
                print(f"=== COCO to YOLO: Copied {idx + 1} of {total_labels} images", end='\r', flush=True)

        print(f"\nFinished copying {copied_count} of {total_labels} images")

    def run(self):
        print(f"Checking if {self.json_path} is a valid COCO annotation file...")
        try:
            COCOVerifier(json_path=args.json_path, fix=False).verify_coco()
        except InvalidCOCOAnnotationError:
            print(f"The provided COCO annotation JSON file is invalid. Exiting...")
            sys.exit(1)
        else:
            print("The provided COCO annotation JSON file is valid.")

        print("Converting COCO annotations to YOLO's format...")
        self.convert()

        print("Done converting. Copying images...")
        self.copy_images()

        print("Success!")

def main(args):
    ConvertCOCOToYOLO(
        img_folder=args.img_folder,
        json_path=args.json_path,
        output_path=args.output_path,
        remove_empty=args.remove_empty,
        keep_duplicates=args.keep_duplicate_labels
    ).run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO format annotations to YOLO format")
    parser.add_argument("img_folder", help="Path to the image folder")
    parser.add_argument("json_path", help="Path to the COCO JSON file")
    parser.add_argument("output_path", help="Path to the output folder")
    parser.add_argument("--remove-empty", action="store_true", default=False, help="Remove images with no annotations in the YOLO format (default: False)")
    parser.add_argument("--keep-duplicate-labels", action="store_true", default=False, help="Keep duplicate labels within an image (default: False)")
    args = parser.parse_args()
    print(args)

    main(args)