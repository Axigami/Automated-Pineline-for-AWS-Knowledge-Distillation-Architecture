#!/bin/bash

# Setup CORS Proxy Server for MLOps

echo "🔧 Setting up CORS proxy server..."

# Install dependencies
echo "📦 Installing dependencies..."
npm install --prefix . express cors node-fetch

# Create .env.local if not exists
if [ ! -f .env.local ]; then
  echo "📝 Creating .env.local..."
  cat > .env.local << EOF
# Local development with CORS proxy
VITE_API_GATEWAY_URL=http://localhost:3001
VITE_SUPABASE_URL=https://zpmbvtfptddmbxhmzapz.supabase.co
VITE_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpwbWJ2dGZwdGRkbWJ4aG16YXB6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQxMDE4MTYsImV4cCI6MjA4OTY3NzgxNn0.jdmdDLKf2xEkb9pJIl-Mc3MgJD_BttQieiknNhr6cT8
EOF
  echo "✅ Created .env.local"
else
  echo "⚠️  .env.local already exists, skipping..."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Start proxy server: node proxy-server.js"
echo "  2. Start frontend: npm run dev"
echo "  3. Test deployment in MLOps page"
echo ""
echo "🔍 The proxy will forward requests from localhost:3001 to API Gateway"
echo "   This bypasses CORS restrictions during development"
