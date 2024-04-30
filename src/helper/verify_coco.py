import argparse
import json

class InvalidCOCOAnnotationError(Exception):
  def __init__(self, message):
    self.message = message
    print(self.message)
    super().__init__(self.message)

class COCOVerifier:
  def __init__(self, json_path, fix):
    self.json_path = json_path
    self.fix = fix

  def check_and_fix_ids(self, coco_json, id_field, section_name, fix_ids=False):
    ids = set(item[id_field] for item in coco_json[section_name])
    sorted_ids = sorted(ids)
    if sorted_ids[0] != 1:
      if fix_ids:
        print(f"Fixing IDs in '{section_name}' to start from 1...")
        id_mapping = {old_id: i + 1 for i, old_id in enumerate(sorted(ids))}
        for item in coco_json[section_name]:
          item[id_field] = id_mapping[item[id_field]]
      else:
        raise InvalidCOCOAnnotationError(f"Not all IDs in '{section_name}' start from 1.")
    # missing_numbers = set()
    # idx = 1
    # for id in sorted_ids:
    #   if id != idx:
    #     missing_numbers.add(idx)
    #     idx += 1
    #   idx += 1
    # if missing_numbers:
    #   if fix_ids:
    #     print(f"Missing IDs in '{section_name}': {missing_numbers}. Fixing...")
    #     # Fix missing IDs
    #     for item in coco_json[section_name]:
    #       if item[id_field] in missing_numbers:
    #         while idx in ids:
    #           idx += 1
    #         item[id_field] = idx
    #         missing_numbers.remove(idx)
    #   else:
    #     raise InvalidCOCOAnnotationError(f"IDs not sequential. Missing IDs in '{section_name}': {missing_numbers}.")

      if fix_ids:
        if section_name == 'images':
          # Change image_id in annotations as well
          for annotation in coco_json['annotations']:
            if annotation['image_id'] in id_mapping:
              annotation['image_id'] = id_mapping[annotation['image_id']]
        elif section_name == 'categories':
          # Change category_id in annotations as well
          for annotation in coco_json['annotations']:
            if annotation['category_id'] in id_mapping:
              annotation['category_id'] = id_mapping[annotation['category_id']]
        return coco_json

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
      required_ann_keys = {'id', 'image_id', 'category_id'}
      if not set(annotation.keys()) >= required_ann_keys:
        raise InvalidCOCOAnnotationError("Invalid annotation format. Each annotation requires at least the following keys: 'id', 'image_id', 'category_id'")

    # Check and fix starting ID for each section
    try:
      result = self.check_and_fix_ids(coco_json, 'id', 'images', self.fix)
      if result: coco_json = result
      result = self.check_and_fix_ids(coco_json, 'id', 'annotations', self.fix)
      if result: coco_json = result
      result = self.check_and_fix_ids(coco_json, 'id', 'categories', self.fix)
      if result: coco_json = result
    except InvalidCOCOAnnotationError as e:
      raise e
    
    # Check for phantom IDs for image_id
    # TODO: Write fix for this
    self.check_phantom_ids(coco_json, 'image_id', 'images', 'annotations')

    # Check for phantom IDs for category_id
    # TODO: Write fix for this
    self.check_phantom_ids(coco_json, 'category_id', 'categories', 'annotations')

    if self.fix:
      # Save the modified JSON to a new file
      fixed_json_path = self.json_path.replace('.json', '_fixed.json')
      with open(fixed_json_path, 'w') as fixed_file:
        json.dump(coco_json, fixed_file)
        print(f"Fixed COCO JSON written to '{fixed_json_path}'.")


def main(args):
  try:
    COCOVerifier(json_path=args.json_path, fix=args.fix).verify_coco()
  except InvalidCOCOAnnotationError:
    print(f"The provided COCO annotation JSON file is invalid.")
  else:
    print("The provided COCO annotation JSON file is valid.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Verifies if COCO Annotations file is valid")
  parser.add_argument("json_path", help="Path to the COCO JSON file")
  parser.add_argument("--fix", action="store_true", help="Attempt to fix the COCO JSON if it is invalid and write to a new file")
  args = parser.parse_args()

  main(args)
