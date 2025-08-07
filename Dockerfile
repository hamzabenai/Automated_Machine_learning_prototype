FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
# Copy ALL files to /app in container (including src/)
COPY . .
CMD ["streamlit", "run", "src/app.py"]
