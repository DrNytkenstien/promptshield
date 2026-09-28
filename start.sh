#!/bin/bash

# 1. Start FastAPI backend in the background on port 8000
uvicorn api.main:app --host 127.0.0.1 --port 8000 &

# 2. Start Streamlit frontend on Render's assigned $PORT
streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0