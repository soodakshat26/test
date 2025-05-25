import os
from PIL import Image, UnidentifiedImageError

img_dir = "images"

for filename in os.listdir(img_dir):
    path = os.path.join(img_dir, filename)
    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, IOError, SyntaxError):
        print(f"Deleting invalid image: {path}")
        os.remove(path)
