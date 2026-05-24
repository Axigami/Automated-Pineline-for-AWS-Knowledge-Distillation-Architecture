@echo off
REM Setup CORS Proxy Server for MLOps (Windows)

echo Setting up CORS proxy server...
echo.

REM Install dependencies
echo Installing dependencies...
call npm install express cors node-fetch
echo.

REM Create .env.local if not exists
if not exist .env.local (
  echo Creating .env.local...
  (
    echo # Local development with CORS proxy
    echo VITE_API_GATEWAY_URL=http://localhost:3001
    echo VITE_SUPABASE_URL=https://zpmbvtfptddmbxhmzapz.supabase.co
    echo VITE_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpwbWJ2dGZwdGRkbWJ4aG16YXB6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQxMDE4MTYsImV4cCI6MjA4OTY3NzgxNn0.jdmdDLKf2xEkb9pJIl-Mc3MgJD_BttQieiknNhr6cT8
  ) > .env.local
  echo Created .env.local
) else (
  echo .env.local already exists, skipping...
)

echo.
echo Setup complete!
echo.
echo Next steps:
echo   1. Start proxy server: node proxy-server.js
echo   2. Start frontend: npm run dev
echo   3. Test deployment in MLOps page
echo.
echo The proxy will forward requests from localhost:3001 to API Gateway
echo This bypasses CORS restrictions during development
echo.
pause
