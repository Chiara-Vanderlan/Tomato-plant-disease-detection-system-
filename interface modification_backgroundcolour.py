import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Function to prepare the image for prediction
def prepare(filepath):
    IMG_SIZE = 70  
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  

# Load the trained model
model = tf.keras.models.load_model("tomatoes1.h5")
CATEGORIES = ["Early Blight", "Late Blight", "Healthy"]


# Function to perform prediction and update the label text
def predict_image():
    filepath = filedialog.askopenfilename()
    if filepath:
        # Load and display the image
        image = Image.open(filepath)
        # Resize the image
        image = image.resize((200, 200))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
        
        image = tf.keras.preprocessing.image.load_img(filepath, target_size=(256, 256))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = tf.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)
        most_likely_class = CATEGORIES[np.argmax(prediction)]
        # label.config(text="Predicted Class: " + most_likely_class)
        # Update the predicted class label
        predicted_label.config(text="Predicted Class: " + most_likely_class, font=("Arial", 12))
        
        # Show the predicted class label
        predicted_label.pack(pady=10)
        
    else:
        messagebox.showerror("Error", "No image selected.")


#Using Tkinter for interface preparation

# Create a Tkinter window
window = tk.Tk()
window.title("Tomato plant leaf disease detection system")

# Set the background color to light green
#window.configure(background='#006400')

# Load the image
background_image = Image.open("background1.jpeg")
background_photo = ImageTk.PhotoImage(background_image)
# Create a label with the background image
background_label = tk.Label(window, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


# Set the initial size of the window
window.geometry("460x460")  # Width x Height
 

# Add instruction text
instruction_label = tk.Label(window, text="Tomato Plant Leaf Disease Detection System", font=("Arial",15), bg="dark green", fg="white")
instruction_label.pack(fill=tk.X, padx=20, pady=(8, 10))
#instruction_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
# Create a button to select an image
select_button = tk.Button(window, text="Select the image:", font=("Arial", 13), command=predict_image)
select_button.pack(fill=tk.X, padx=160, pady=8)

# Create a label to display the predicted class (initially hidden)
predicted_label = tk.Label(window, text="")

# # Create a label to display the uploaded image
image_label = tk.Label(window)
image_label.pack(pady=10)
#image_label.pack_forget() 



window.mainloop()
