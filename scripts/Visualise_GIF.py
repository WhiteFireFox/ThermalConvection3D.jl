from PIL import Image
import os

# Define folder path and image sequence
folder_path = './output/'  # Replace with the path to your images folder
filenames = [f"T_3D_{str(i).zfill(4)}.png" for i in range(0,99)]

# Load images
images = [Image.open(os.path.join(folder_path, filename)) for filename in filenames]

# Create GIF
output_gif_path = 'output.gif'  # The path where you want to save the GIF
images[0].save(output_gif_path, save_all=True, append_images=images[1:], loop=0, duration=100)

print(f"GIF saved at {output_gif_path}")
