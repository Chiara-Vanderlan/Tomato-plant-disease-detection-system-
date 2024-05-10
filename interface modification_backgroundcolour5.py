#Image segementation included

import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk



# Function to prepare the image for prediction
def prepare(filepath):
    IMG_SIZE = 256
    
    # Read the image in color
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    # Convert the image from BGR to HSV color space
    hsv_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
    
    # Define range of green color in HSV
    lower_green = np.array([25, 52, 72])  # Lower bound of green
    upper_green = np.array([102, 255, 255])  # Upper bound of green
    
    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv_array, lower_green, upper_green)
    
    # Bitwise-AND mask and original image
    segmented_img = cv2.bitwise_and(img_array, img_array, mask=mask)
    
    # Apply Gaussian Blur to reduce noise
    segmented_img = cv2.GaussianBlur(segmented_img, (5, 5), 0)
    
    # Resize the image to the desired size
    new_array = cv2.resize(segmented_img, (IMG_SIZE, IMG_SIZE))
    
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # Change channel dimension to 3 as we're using color images now


# Load the trained model
model = tf.keras.models.load_model("tomatoes1.h5")
CATEGORIES = ["Early Blight", "Late Blight", "Healthy"]


# Function to perform prediction and update the label text
def predict_image():
    filepath = filedialog.askopenfilename()
    if filepath:
        # Load and display the original image
        original_image = Image.open(filepath)
        original_image = original_image.resize((256, 256))
        original_photo = ImageTk.PhotoImage(original_image)
        # Create a label to display the uploaded image
        image_label = tk.Label(window)
        image_label.pack(pady=10)

        image_label.config(image=original_photo)
        image_label.image = original_photo
        
        # Perform image processing steps
        processed_image = prepare(filepath)
        processed_image = processed_image.reshape(256, 256, 3)  # Reshape for display
        
        # Display the processed image
        processed_photo = ImageTk.PhotoImage(image=Image.fromarray(processed_image))
        processed_label.config(image=processed_photo)
        processed_label.image = processed_photo
        processed_label.pack(pady=10)
        
        # Update the label text with predicted class
        image = tf.keras.preprocessing.image.load_img(filepath, target_size=(256, 256))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = tf.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)
        most_likely_class = CATEGORIES[np.argmax(prediction)]
        predicted_label.config(text="Predicted Class: " + most_likely_class, font=("Arial", 12))
        
        # Show the predicted class label
        predicted_label.pack(pady=10)
        
    else:
        messagebox.showerror("Error", "No image selected.")

# Create a Tkinter window
window = tk.Tk()
window.title("Tomato plant leaf disease detection system")

# Load the image
background_image = Image.open("background1.jpeg")
background_photo = ImageTk.PhotoImage(background_image)
# Create a label with the background image
background_label = tk.Label(window, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Set the initial size of the window
window.geometry("460x760")  # Width x Height

# Add instruction text
instruction_label = tk.Label(window, text="Tomato Plant Leaf Disease Detection System", font=("Arial",15), bg="dark green", fg="white")
instruction_label.pack(fill=tk.X, padx=20, pady=(8, 10))

# Create a button to select an image
select_button = tk.Button(window, text="Select the image:", font=("Arial", 13), command=predict_image)
select_button.pack(fill=tk.X, padx=160, pady=8)

# Create labels to display the processed image and predicted class
processed_label = tk.Label(window)
processed_label.pack(pady=10)
predicted_label = tk.Label(window, text="")
predicted_label.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()
