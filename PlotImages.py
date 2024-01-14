from PIL import Image, ImageDraw
import os

# Define the directories where your images are located
base_dir = "./dataset"  # Replace with the actual path


def compute_image(base_dir, name) :
    # List of personality directories (person1, person2, ..., person5)
    personality_directories = [f"person{i}" for i in range(1, 6)]
    # Loop through each personality directory
    for personality_dir in personality_directories:
        dir = os.path.join(base_dir, personality_dir, name)

        # Create a list to store reference and test images
        images_to_combine = []
        # Print personality name
        print(f"Personality: {personality_dir}")
        idx_start = 0;
        idx_end = 0
        if name == "test" :
            idx_start = 8
            idx_end = 10
        else :
            idx_start = 1
            idx_end = 8

        # Process reference images
        for i in range(idx_start, idx_end):
            image_path = os.path.join(dir, f"image{i}.jpg")
            if os.path.exists(image_path):
                img = Image.open(image_path)
                img = img.resize((200, 200))  # Adjust the size as needed
                images_to_combine.append(img)


        # Calculate the width and height of the combined image
        total_width = len(images_to_combine) * 200
        max_height = 200

        # Create a new image to combine all images
        combined_image = Image.new('RGB', (total_width, max_height))

        # Paste each image onto the combined image
        x_offset = 0
        for img in images_to_combine:
            combined_image.paste(img, (x_offset, 0))
            x_offset += 200

        # Save or show the combined image
        combined_image.show()

        # Optionally, you can save the combined image to a file
        combined_image.save(os.path.join(base_dir, personality_dir, f"{name}combined_image.jpg"))

# compute_image(base_dir, "reference")
compute_image(base_dir, "reference")


