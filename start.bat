@echo off
echo Starting ML-TEST Platform...

echo Starting backend (FastAPI)...
cd backend
if not exist .venv (
    echo Creating Python 3.12 virtual environment...
    "C:\Users\zink\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv
    .venv\Scripts\pip install -r requirements.txt -q
)
start "ML-TEST Backend" .venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
cd ..

echo Starting frontend (Vite + React)...
cd frontend
if not exist node_modules npm install --silent
start "ML-TEST Frontend" npm run dev
cd ..

echo.
echo ML-TEST is running!
echo   Frontend: http://localhost:5173
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo.
pause
