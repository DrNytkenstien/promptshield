#!/bin/bash
# Start FastAPI backend in the background on port 8000
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit on Render's assigned $PORT
streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0