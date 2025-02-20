import os
import re
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def resize_and_place_in_circle(img, circle_radius, circle_position, image_size, scale_factor=0.7, image_offset=-20):
    # Resize the image to fit within the circle
    scale = min(circle_radius * 2 / img.size[0], circle_radius * 2 / img.size[1]) * scale_factor
    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Create a white background
    background = Image.new('RGBA', image_size, (255, 255, 255, 255))
    # Position the image in the center
    img_position = (circle_position[0] - new_size[0] // 2, circle_position[1] - new_size[1] // 2 + image_offset)
    background.paste(img, img_position, img)

    # Create a mask for the circle and apply blur
    circle_mask = Image.new('L', image_size, 0)
    draw_mask = ImageDraw.Draw(circle_mask)
    draw_mask.ellipse((circle_position[0] - circle_radius, circle_position[1] - circle_radius, 
                       circle_position[0] + circle_radius, circle_position[1] + circle_radius), 
                      fill=255)
    blur_radius = 2
    circle_mask = circle_mask.filter(ImageFilter.GaussianBlur(blur_radius))
    background.putalpha(circle_mask)

    return background

def resize_and_process_image(file_path, font_path):
    with Image.open(file_path) as img:
        # Convert to RGBA if necessary
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        circle_radius = 128
        circle_position = (128, 128)  # Circle positioned slightly lower
        image_size = (256, 256)

        # Resize and place original image in a circle
        object_name = re.search(r'(\w+)_icon\.png$', file_path).group(1)
        if object_name in ["fish", "insect", "shark", "turtle", "crab", "dog", "circle", "triangle", "lizard", "monkey"]:
            scale_factor = 0.9
        elif object_name in ["rodent", "snake", "frog"]:
            scale_factor = 0.8
        elif object_name in ["house_cat"]:
            scale_factor = 0.6
        elif object_name in ["fruit", "fungus", "vegetable"]:
            scale_factor = 0.65
        else:
            scale_factor = 0.7

        if object_name in ["vegetable"]:
            image_offset = -40
        elif object_name in ["house_cat", "fungus", "antelope", "rodent"]:
            image_offset = -30
        else:
            image_offset = -20

        final_img = resize_and_place_in_circle(img, circle_radius, circle_position, image_size, scale_factor, image_offset)

        # Add text
        if object_name in ["house_cat", "vegetable"]:
            text_offset = -45
        elif object_name in ["antelope"]:
            text_offset = -35
        else:
            text_offset = -30

        draw = ImageDraw.Draw(final_img)
        font = ImageFont.truetype(font_path, 36)
        object_name = object_name.replace("_", " ")
        text_width, text_height = draw.textsize(object_name, font=font)
        draw.text(((image_size[0] - text_width) / 2, image_size[1] - text_height + text_offset), object_name, font=font, fill="black")

        return final_img

def process_images(directory, font_path):
    # Ensure the edited_icons directory exists
    os.makedirs('edited_icons', exist_ok=True)

    for filename in os.listdir(directory):
        if filename.endswith('.png') and 'icon' in filename:
            file_path = os.path.join(directory, filename)
            final_img = resize_and_process_image(file_path, font_path)
            new_file_path = os.path.join('edited_icons', filename)
            final_img.save(new_file_path, 'PNG')

# Run the function for the current directory
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
process_images('.', font_path)
