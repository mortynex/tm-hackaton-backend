FROM python:3.9  # Or another suitable Python base image

# Install required packages
RUN pip install transformers==4.35.2 diffusers==0.25.0 accelerate==0.25.0 opencv-fixer==0.2.5 flask flask_cors

# Fix known OpenCV issues if needed
RUN python -c "from opencv_fixer import AutoFix; AutoFix()"  

# Set working directory
WORKDIR /backend

# Copy your Python code 
COPY *.py /backend/  

# Copy requirements.txt if you have one
COPY requirements.txt /backend/ 
RUN pip install -r requirements.txt


# Specify the command to start your Flask backend
CMD ["python", "backend.py"]  # Assuming your main Python file is named backend.py
