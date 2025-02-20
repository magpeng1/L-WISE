from PIL import Image, ImageDraw, ImageFont, ImageFilter

def generate_image_with_transparent_bg_and_blurred_circle(word):
    image_size = (256, 256)
    circle_color = '#e7e8e9'
    font_color = 'black'
    font_size = 40
    font_path = "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
    line_spacing = 8  # Space between lines

    img = Image.new('RGBA', image_size, color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)

    circle_x = image_size[0] // 2
    circle_y = image_size[1] // 2
    circle_radius = 126

    draw.ellipse((circle_x - circle_radius, circle_y - circle_radius, 
                  circle_x + circle_radius, circle_y + circle_radius), 
                 fill=circle_color)

    blur_radius = 2
    circle_mask = Image.new('L', image_size, 0)
    draw_mask = ImageDraw.Draw(circle_mask)
    draw_mask.ellipse((circle_x - circle_radius, circle_y - circle_radius, 
                       circle_x + circle_radius, circle_y + circle_radius), 
                      fill=255)
    circle_mask = circle_mask.filter(ImageFilter.GaussianBlur(blur_radius))
    img.putalpha(circle_mask)

    # Split the word into lines
    lines = word.split('\n')
    total_height = (len(lines) - 1) * line_spacing

    # Calculate the total height of the text block
    for line in lines:
        text_height = font_size
        total_height += text_height

    # Calculate the starting Y position
    y = (image_size[1] - total_height) // 2

    # Draw each line
    for line in lines:
        text_width = draw.textlength(line, font=font)
        text_height = font_size
        x = (image_size[0] - text_width) // 2
        draw.text((x, y), line, font=font, fill=font_color)
        y += text_height + line_spacing  # Move y to the next line position

    word_mod = word.replace("/", "_").replace("\n", "_").replace(" ", "_")
    file_path = f'images/{word_mod}_text_icon.png'
    img.save(file_path, format="PNG")
    
    return file_path

# labels = ["Circle\nF", "Triangle\nF", "Circle\nJ", "Triangle\nJ"]
#labels = ["melanoma", "benign\nmole", "basal cell\ncarcinoma", "benign\nkeratosis"]
labels = ["Benign\n(press F)", "Malignant\n(press J)"]


for label in labels:
  file_path = generate_image_with_transparent_bg_and_blurred_circle(label)
  print(file_path)