# Use the official Python slim image
FROM python:3.10-slim

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Expose the application port (not necessary here but good for debugging)
EXPOSE 5001

# Command to run the script
CMD ["python", "fetch_healthEvents.py"]  
