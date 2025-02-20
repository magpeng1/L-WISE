from PIL import Image, ImageDraw, ImageFont, ImageFilter

def generate_image_with_transparent_bg_and_blurred_circle(word):
    # Define the size of the image
    image_size = (256, 256)
    # Define the circle color
    circle_color = '#e7e8e9'  # Grey color from wormholes images
    # Define the font color
    font_color = 'black'
    # Define the font size
    font_size = 42
    # Define the path to the font
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    
    # Create an image with transparent background
    img = Image.new('RGBA', image_size, color=(0, 0, 0, 0))
    # Initialize the drawing context with the image as background
    draw = ImageDraw.Draw(img)
    # Load a font
    font = ImageFont.truetype(font_path, font_size)
    
    # Calculate the position of the circle
    circle_x = image_size[0] // 2
    circle_y = image_size[1] // 2
    circle_radius = 126

    # Draw the circle
    draw.ellipse((circle_x - circle_radius, circle_y - circle_radius, 
                  circle_x + circle_radius, circle_y + circle_radius), 
                 fill=circle_color)

    # Apply blur to the circle
    blur_radius = 2  # The radius of the blur
    circle_mask = Image.new('L', image_size, 0)
    draw_mask = ImageDraw.Draw(circle_mask)
    draw_mask.ellipse((circle_x - circle_radius, circle_y - circle_radius, 
                       circle_x + circle_radius, circle_y + circle_radius), 
                      fill=255)
    circle_mask = circle_mask.filter(ImageFilter.GaussianBlur(blur_radius))
    img.putalpha(circle_mask)

    # Get the width and height of the text to be drawn
    text_width = draw.textlength(word, font=font)
    text_height = font_size

    # Calculate the x, y coordinates of the text
    x = (image_size[0] - text_width) // 2
    y = (image_size[1] - text_height) // 2

    # Draw the text on the image
    draw.text((x, y), word, font=font, fill=font_color)

    # Save the image to a file
    file_path = f'images/{word.replace(" ", "_")}_text_icon.png'
    img.save(file_path, format="PNG")
    
    return file_path

# Example usage

# labels = [
#     "circle",
#     "triangle",
#     "neutrophil",
#     "lymphocyte",
#     "monocyte",
#     "basophil",
#     "eosinophil",
#     "A",
#     "B",
#     "C",
#     "D",
#     "E",
#     "bird",
#     "crab",
#     "furniture",
#     "tool",
#     "boat",
#     "dog",
#     "insect",
#     "truck",
#     "building",
#     "instrument",
#     "turtle",
#     "frog",
#     "lizard",
#     "fruit",
#     "monkey",
#     "clothing",
#     "fungus",
#     "cat",
#     "car",
#     "fish",
#     "fruit",
#     "snake",
# ]

# labels = ["Welshie", "Brittany"]
# labels = ["A", "B", "C", "D", ]
# labels = ["Sandpiper", "Dowitcher", "Totanus", "Oyster-\ncatcher", "Turnstone"] # Totanus ~= redshank
# labels = ["sandpiper", "dowitcher", "totanus", "oystercatcher", "turnstone", "circle", "triangle"] # Totanus ~= redshank
# labels = ["A", "B", "C", "D", "E"]
# labels = ["Ajax", "Tyro", "Leda"]
# labels = ["Circle", "Triangle"]
# labels = ["F", "J"]
# labels = ["Circle\n(press F)", "Triangle\n(press F)"]
#labels = ["Ajax", "Tyro", "Leda", "Eris"]
labels = ["Benign", "Malignant"]

for label in labels:
  file_path = generate_image_with_transparent_bg_and_blurred_circle(label)
