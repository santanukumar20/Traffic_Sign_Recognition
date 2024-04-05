import cv2
import numpy as np
import tensorflow as tf
import time  

# Load the pre-trained model
model_path = "model.h5"
model = tf.keras.models.load_model(model_path)

# Define classes for traffic signs
classes = ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 'No passing', 'No passing veh over 3.5 tons', 'Right-of-way at intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles', 'Veh > 3.5 tons prohibited', 'No entry', 'General caution', 'Dangerous curve left', 'Dangerous curve right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End speed + passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing', 'End no passing veh > 3.5 tons']

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
  
    ret, frame = cap.read()

    # Check if frame is read successfully
    if not ret:
        print("Error: Could not read frame.")
        break
    
    
    
    # Resize and preprocess the image for model input
    processed_frame = cv2.resize(frame, (30, 30))  
    processed_frame = np.expand_dims(processed_frame, axis=0)  # Add batch dimension
    processed_frame = processed_frame / 255.0  # Normalize pixel values
    
   
    predictions = model.predict(processed_frame)
    
    
    class_id = np.argmax(predictions)
    conf = predictions[0, class_id]
    
   
    if conf > 0.5:
        
        x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]  
        
       
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, classes[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    

    cv2.imshow('Traffic Sign Detection', frame)
    
    
    time.sleep(0.1)
    
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
