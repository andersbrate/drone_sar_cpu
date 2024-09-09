# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv

# Install Gunicorn
RUN pip install gunicorn==21.2.0

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Make port 9999 available to the world outside this container
EXPOSE 9999

# Define environment variable
ENV OBJECT_DETECTION_API_URL=http://127.0.0.1:5000/process

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:9999", "app:app"]
