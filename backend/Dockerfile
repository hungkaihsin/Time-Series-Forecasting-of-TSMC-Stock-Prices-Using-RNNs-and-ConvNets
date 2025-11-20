# Use an official Python runtime as a parent image
FROM python:3.9-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY src ./src

# Specify the command to run on container startup
# The application will be served by Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.api.app:app"]
