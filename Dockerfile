# Use a small official Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python deps (add any extras you need)
RUN pip install --no-cache-dir \
    dash \
    dash-table \
    plotly \
    pandas \
    numpy \
    yfinance \
    gunicorn

# Copy your app code into the image
COPY . /app

# Dash listens on 8050 by default
EXPOSE 8050

# Serve via Gunicorn (Dash exposes a Flask server at app.server)
# If your main file isn't app.py, change "app:app.server" to "<module>:app.server"
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "2", "--threads", "4", "app:server"]