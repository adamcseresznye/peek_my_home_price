# Use NiceGUI base image
FROM zauberzeug/nicegui:2.5.0

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app/

# Install the required packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /app

# Set environment variable for Python unbuffered output
ENV PYTHONUNBUFFERED=1

# Create entrypoint script
RUN echo '#!/bin/bash\n\
export PORT="${PORT:-8080}"\n\
exec uvicorn src.ui.main:api --host 0.0.0.0 --port "$PORT" --workers 1 --log-level info\
' > /entrypoint.sh && chmod +x /entrypoint.sh

# Expose the port
EXPOSE 8080

# Use the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
