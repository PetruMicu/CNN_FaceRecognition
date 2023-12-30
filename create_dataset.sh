#!/bin/bash

# Replace these variables with the actual paths
source_directory="./images"
dataset_root="./dataset"

# Create the main dataset directory
mkdir -p "$dataset_root"

# Loop through personalities and create the structure
personalities=("person1" "person2" "person3" "person4" "person5")

for person in "${personalities[@]}"; do
    # Create reference and test directories for each personality
    mkdir -p "$dataset_root/$person/reference"
    mkdir -p "$dataset_root/$person/test"
    
    # Copy the first 7 images to the reference directory
    cp "$source_directory/$person/"{image1.jpg,image2.jpg,image3.jpg,image4.jpg,image5.jpg,image6.jpg,image7.jpg} "$dataset_root/$person/reference/"
    
    # Copy the next 2 images to the test directory
    cp "$source_directory/$person/"{image8.jpg,image9.jpg} "$dataset_root/$person/test/"
done

echo "Dataset structure created successfully."
