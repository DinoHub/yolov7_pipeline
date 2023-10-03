# Resizes all images in the given image dir into the given width and height
import argparse
from PIL import Image
from pathlib import Path
from glob import glob
import shutil

# Define a set of valid image extensions
IMG_EXTS = {".png", ".jpg", ".jpeg"}

def input_dir_is_image_dir(input_dir):
    """
    Check if image files exist in the specified directory or within a 'images' subfolder.

    Args:
        input_dir (Path): The directory path to be checked.

    Returns:
        bool: True if input dir contains images, False if it contains a folder with images.
    """
    # Check if the input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"Invalid input directory: '{input_dir}' does not exist.")

    # Check if there are image files directly in the input directory
    for file in input_dir.iterdir():
        if file.suffix.lower() in IMG_EXTS and file.is_file():
            return True

    # Check if there's a folder called "images" containing images
    images_folder = input_dir / "images"
    if images_folder.exists() and images_folder.is_dir():
        for file in images_folder.iterdir():
            if file.suffix.lower() in IMG_EXTS and file.is_file():
                return False

    raise Exception(f"Invalid input directory: '{input_dir}' should contain images (.png, .jpg, .jpeg) or contain 'images' folder with images.")

def create_or_clear_folder(folder):
    """
    Create the specified folder if it doesn't exist, or clear its contents.

    Args:
        folder (str or Path): Path to the folder.
    """
    folder = Path(folder)

    try:
        shutil.rmtree(folder)
    except FileNotFoundError:
        print(f"Output folder {folder} does not exist, creating folder...")
    except Exception as e:
        print(f"An error occurred while deleting {folder}: {e}")
    else:
        print(f"Deleted existing contents in folder: {folder}")

    folder.mkdir(parents=True, exist_ok=True)

def main(args):
    input_image_dir = Path(args.input_dir)
    output_image_dir = Path(args.output_dir)
    image_folder_is_subdirectory = not input_dir_is_image_dir(input_image_dir)
    if image_folder_is_subdirectory:
        input_image_dir = input_image_dir / "images"
        output_image_dir = output_image_dir / "images"

    create_or_clear_folder(output_image_dir)

    print(f"Starting image resizing for {input_image_dir}")
    image_files = [file for file in input_image_dir.iterdir() if file.suffix.lower() in IMG_EXTS]
    total_images = len(image_files)

    for i, img in enumerate(image_files):
        image = Image.open(img)
        image.thumbnail((args.width, args.height))
        output_path = output_image_dir / img.name
        image.save(str(output_path))

        if (i + 1) % 100 == 0 or (i + 1) == total_images:
            print(f"=== Image resizing: Processed {i + 1}/{total_images} images", end='\r', flush=True)
    
    if image_folder_is_subdirectory:
        try:
            input_label_dir = Path(args.input_dir) / "labels"
        except FileNotFoundError:
            print(f"\nWARNING: Labels folder does not exist in {args.input_dir}. Output dir {args.output_dir} only consists of 'images' folder.")
        else:
            print(f"\nCompleted resizing of images. Copying labels into {str(input_label_dir)}")

            output_label_dir = Path(args.output_dir) / "labels"
            create_or_clear_folder(output_label_dir)

            for source_file_path in input_label_dir.glob("*.txt"):
                destination_file_path = output_label_dir / source_file_path.name
                shutil.copy2(source_file_path, destination_file_path)

        print(f"Completed. Output dir: {args.output_dir}")

    else:
        print(f"\nCompleted. Output image dir: {str(output_image_dir)}")


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Resize images")
    parser.add_argument("--input-dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output image directory")
    parser.add_argument("--width", type=int, default=1280, help="Resized width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Resized height (default: 720)")
    args = parser.parse_args()
    print(args)

    main(args)