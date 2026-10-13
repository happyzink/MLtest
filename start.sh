#!/bin/bash
# ML-TEST Startup Script

echo "🚀 Starting ML-TEST Platform..."

# Start backend
echo "📦 Starting backend (FastAPI)..."
cd backend
if [ ! -d ".venv" ]; then
    echo "Creating Python 3.12 virtual environment..."
    "/c/Users/zink/AppData/Local/Programs/Python/Python312/python.exe" -m venv .venv
    .venv/Scripts/pip install -r requirements.txt -q
fi
.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

# Start frontend
echo "🎨 Starting frontend (Vite + React)..."
cd frontend
[ ! -d "node_modules" ] && npm install --silent
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ ML-TEST is running!"
echo "   Frontend: http://localhost:5173"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
