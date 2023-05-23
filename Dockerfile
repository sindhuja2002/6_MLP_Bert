FROM python:3.9

WORKDIR /app

# Install dependencies
RUN pip install torch transformers

# Copy the Python script into the container
COPY generate_titles.py .

# Set the entry point
ENTRYPOINT ["python", "generate_titles.py"]
