#!/bin/bash
# Script to start the CAMEL Extensions application with monitoring enabled

# Set environment variables for monitoring
export LOG_LEVEL=INFO
export PROMETHEUS_PORT=9090
export ENABLE_PROMETHEUS=true

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to handle cleanup on exit
cleanup() {
    echo "Shutting down services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up trap for cleanup
trap cleanup SIGINT SIGTERM

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python 3.10 or 3.11 (as recommended in README.md)."
    exit 1
fi

# Check if required directories exist
if [ ! -d "backend" ] || [ ! -d "gui" ]; then
    echo "Error: backend or gui directory not found. Please run this script from the project root."
    exit 1
fi

# Install all dependencies if needed
if [ ! -f ".deps_installed" ]; then
    echo "Installing project dependencies..."
    pip install -r requirements.txt
    touch .deps_installed
fi

# Start the backend server
echo "Starting backend server with monitoring enabled..."
cd backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 5

# Start the frontend
echo "Starting frontend..."
cd gui
streamlit run app.py &
FRONTEND_PID=$!
cd ..

echo "Services started:"
echo "- Backend: http://localhost:8000"
echo "- Frontend: http://localhost:8501"
echo "- Prometheus metrics: http://localhost:9090"
echo "- API metrics: http://localhost:8000/api/metrics"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for processes to finish
wait $BACKEND_PID $FRONTEND_PID