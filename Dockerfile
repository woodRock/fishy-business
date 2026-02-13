# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for some python packages like pygraphviz or opencv if used later)
RUN apt-get update && apt-get install -y 
    build-essential 
    graphviz 
    libgraphviz-dev 
    git 
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Expose ports for Streamlit (8501) or other services
EXPOSE 8501

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run the dashboard by default
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
