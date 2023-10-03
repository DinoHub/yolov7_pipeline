import argparse
import json

class InvalidCOCOAnnotationError(Exception):
    pass

class VerifyCOCO:
    def __init__(self, json_path):
        self.json_path = json_path

    def check_phantom_ids(self, coco_json, id_field, reference_key, target_key):
        reference_ids = set(item['id'] for item in coco_json[reference_key])
        target_ids = set(item[id_field] for item in coco_json[target_key])

        phantom_ids = target_ids - reference_ids

        if phantom_ids:
            print(f"Phantom {id_field}s found in '{target_key}': {phantom_ids}")
            raise InvalidCOCOAnnotationError(f"Invalid annotations: Phantom IDs (exists in {target_key} but does not exist in {reference_key}).")

    def verify_coco(self):
        print(f"Running verification on {self.json_path}...")

        try:
            with open(self.json_path, 'r') as file:
                coco_json = json.load(file)
        except FileNotFoundError:
            print(f"File '{self.json_path}' not found.")
        except json.JSONDecodeError:
            print(f"Error parsing JSON in file '{self.json_path}'.")

        # Check if coco_json is a dictionary
        if not isinstance(coco_json, dict):
            raise InvalidCOCOAnnotationError("Invalid JSON structure: Expected a dictionary.")

        # Check if required keys are present
        required_keys = {'images', 'annotations', 'categories'}
        if not required_keys.issubset(set(coco_json.keys())):
            raise InvalidCOCOAnnotationError(f"Missing required keys in COCO annotation file. File only has {coco_json.keys()} but requires {required_keys}")

        # Check if 'images', 'annotations', and 'categories' are lists
        for key in ['images', 'annotations', 'categories']:
            if not isinstance(coco_json[key], list):
                raise InvalidCOCOAnnotationError(f"'{key}' should be a list.")

        # Check if each annotation entry has required fields
        for annotation in coco_json['annotations']:
            required_ann_keys = {'id', 'image_id', 'category_id', 'bbox', 'area', 'iscrowd', 'segmentation'}
            if not set(annotation.keys()) == required_ann_keys:
                raise InvalidCOCOAnnotationError("Invalid annotation format.")
        
        # Check if the IDs start from 1
        for key in ['images', 'annotations', 'categories']:
            ids = set(item['id'] for item in coco_json[key])
            if not all(id == i + 1 for i, id in enumerate(sorted(ids))):
                raise InvalidCOCOAnnotationError(f"Not all IDs in '{key}' start from 1.")
        
        # Check for phantom IDs for image_id
        self.check_phantom_ids(coco_json, 'image_id', 'images', 'annotations')

        # Check for phantom IDs for category_id
        self.check_phantom_ids(coco_json, 'category_id', 'categories', 'annotations')

def main(args):
    try:
        VerifyCOCO(json_path=args.json_path).verify_coco()
    except InvalidCOCOAnnotationError:
        print(f"The provided COCO annotation JSON file is invalid.")
    else:
        print("The provided COCO annotation JSON file is valid.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifies if COCO Annotations file is valid")
    parser.add_argument("json_path", help="Path to the COCO JSON file")
    args = parser.parse_args()

    main(args)
