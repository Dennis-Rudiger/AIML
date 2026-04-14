import os
import random
from PIL import Image, ImageDraw

def create_shape_image(shape_type, img_size=(64, 64), bg_color=(0,0,0)):
    """
    Creates a simple image of a circle or square with a random color.
    Uses PIL (Python Imaging Library).
    """
    img = Image.new('RGB', img_size, color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Pick a random bright color for the shape
    shape_color = (random.randint(50,255), random.randint(50,255), random.randint(50,255))
    
    # Define a random bounding box inside the image (pad by 5 pixels)
    padding = 5
    w, h = img_size
    
    # We create random sizes and positions to make the dataset robust (data variation!)
    x0 = random.randint(padding, w//2 - padding)
    y0 = random.randint(padding, h//2 - padding)
    x1 = random.randint(w//2 + padding, w - padding)
    y1 = random.randint(h//2 + padding, h - padding)
    
    if shape_type == 'circle':
        draw.ellipse([x0, y0, x1, y1], fill=shape_color)
    elif shape_type == 'square':
        draw.rectangle([x0, y0, x1, y1], fill=shape_color)
        
    return img

def generate_dataset(base_dir='data', samples_per_class=500, test_split=0.2):
    """
    Generates a folder structure required by PyTorch ImageFolder.
    Format:
    data/
      train/
        circle/ (e.g., 400 images)
        square/ (e.g., 400 images)
      test/
        circle/ (e.g., 100 images)
        square/ (e.g., 100 images)
    """
    classes = ['circle', 'square']
    splits = ['train', 'test']
    
    # Ensure working inside the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, base_dir)
    
    print(f"Generating dataset at: {data_dir}")
    print(f"Total samples per class: {samples_per_class}")
    
    # Create the folder structure
    for split in splits:
        for cls in classes:
            folder_path = os.path.join(data_dir, split, cls)
            os.makedirs(folder_path, exist_ok=True)
            
    # Generate the images
    for cls in classes:
        print(f"Creating images for class: {cls}...")
        for i in range(samples_per_class):
            img = create_shape_image(cls)
            
            # Decide if this goes into training (80%) or testing (20%)
            if random.random() < test_split:
                save_dir = os.path.join(data_dir, 'test', cls)
            else:
                save_dir = os.path.join(data_dir, 'train', cls)
                
            file_path = os.path.join(save_dir, f"{cls}_{i}.png")
            img.save(file_path)
            
    print("\nDataset generation complete!")
    print("Now you can train the Deep Learning CNN Model.")

if __name__ == "__main__":
    generate_dataset()
