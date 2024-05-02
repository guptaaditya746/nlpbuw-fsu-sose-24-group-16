# Use a base image that has Python and necessary libraries for the model
FROM fschlatt/natural-language-processing-exercises:0.0.1

# Add the script for making predictions
ADD run.py /code/run.py

# Add the trained model file
ADD model.joblib /code/model.joblib

# Set the working directory
WORKDIR /code

# Specify the entry point for the Docker container
ENTRYPOINT ["python3", "run.py"]
