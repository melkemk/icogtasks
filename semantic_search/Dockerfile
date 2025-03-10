# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Set the working directory to /app/src
WORKDIR /app/src

# Run the application
CMD ["python", "app.py"]