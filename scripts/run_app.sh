#!/bin/bash

# Activate virtual environment (if using one)
# source /path/to/your/venv/bin/activate

# Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
