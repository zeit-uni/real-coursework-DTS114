# Use the lightweight official Python 3.11 image as the base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the local requirements.txt to the working directory of the container
COPY requirements.txt .

# Run the pip command inside the container to install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files in the project to the working directory of the container
COPY . .

# Declare the ports that the container will listen to at runtime
EXPOSE 5005

# Commands executed when the container starts
CMD ["gunicorn", "--bind", "0.0.0.0:5005", "main:app"]