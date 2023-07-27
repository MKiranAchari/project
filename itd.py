from PIL import Image
import io
import numpy as np

def image_to_data(image_path):
    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Convert the image to RGB mode (if it's in a different mode)
            img = img.convert("RGB")
            
            # Convert the image to bytes
            img_byte_array = io.BytesIO()
            img.save(img_byte_array, format="PNG")
            img_bytes = img_byte_array.getvalue()
            
            # Convert the bytes to a numpy array
            img_np_array = np.frombuffer(img_bytes, dtype=np.uint8)
            
            return img_np_array
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    try:
        # Get the image path from the user
        image_path = "E:/img/7.jpg"

        # Convert the image to data
        image_data = image_to_data(image_path)

        if image_data is not None:
            print(f"Image data: {image_data}")
    except KeyboardInterrupt:
        print("\nOperation aborted by the user.")

if __name__ == "_main_":
    main()
