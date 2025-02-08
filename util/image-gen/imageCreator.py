from dotenv import load_dotenv
from openai import OpenAI
import os
from PIL import Image
import io

# Load the .env file
load_dotenv()

# Get your OpenAI API key from the environment variable
api_key = os.getenv("API_KEY")

if api_key is None:
    raise ValueError("API key not found. Please set OPENAI_API_KEY in the .env file.")

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=api_key)

# Function to resize, compress and convert the image format
def resize_image(image_path, max_size_mb=3):
    # Open the image
    with Image.open(image_path) as img:
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Save to a bytes buffer to check the size
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')

        # Check the current size
        current_size_mb = len(img_byte_arr.getvalue()) / (1024 * 1024)

        # If it's larger than the max_size_mb, resize and compress
        if current_size_mb > max_size_mb:
            print(f"Image is {current_size_mb:.2f} MB, resizing and compressing...")

            # Calculate scale factor to reduce the size
            scale_factor = (max_size_mb / current_size_mb) ** 0.5  # Scale factor based on area

            # Calculate new dimensions
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)

            # Resize the image using LANCZOS resampling
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save the resized image to a bytes buffer with reduced compression
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG', optimize=True, quality=85)  # Optimize for smaller size

        # Check the size after resizing
        resized_size_mb = len(img_byte_arr.getvalue()) / (1024 * 1024)
        print(f"Resized image size: {resized_size_mb:.2f} MB")

        # Return the resized image data
        return img_byte_arr.getvalue()

# Provide the image path
image_path = "testImages/test1.png"  # Replace with your image path

# Resize and compress the image if necessary
image_data = resize_image(image_path)

response = client.images.edit(
    model="dall-e-2",
    image=image_data,
    #mask=open("mask.png", "rb"),
    prompt="Turn this image into a caricature with exaggerated facial features: large eyes, big head, and small body. Add a colorful background with playful elements.",
    n=1,
    size="1024x1024",
)

print(response.data[0].url)
