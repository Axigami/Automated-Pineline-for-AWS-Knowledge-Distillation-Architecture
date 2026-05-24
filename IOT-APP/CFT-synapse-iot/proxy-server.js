/**
 * CORS Proxy Server
 * 
 * This proxy server forwards requests from localhost to API Gateway
 * and adds CORS headers to bypass browser CORS restrictions.
 * 
 * Usage:
 *   node proxy-server.js
 * 
 * Then update .env:
 *   VITE_API_GATEWAY_URL=http://localhost:3001
 */

const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');

const app = express();
const PORT = 3001;
const API_GATEWAY_URL = 'https://fbujw415e6.execute-api.ap-southeast-2.amazonaws.com/prod';

// Enable CORS for all origins
app.use(cors());
app.use(express.json());

// Proxy all requests to API Gateway
app.all('*', async (req, res) => {
  const targetUrl = `${API_GATEWAY_URL}${req.path}`;
  
  console.log(`[Proxy] ${req.method} ${req.path} → ${targetUrl}`);
  
  try {
    const response = await fetch(targetUrl, {
      method: req.method,
      headers: {
        'Content-Type': 'application/json',
        ...(req.headers.authorization && { 'Authorization': req.headers.authorization }),
      },
      body: req.method !== 'GET' ? JSON.stringify(req.body) : undefined,
    });
    
    const data = await response.json();
    
    console.log(`[Proxy] Response status: ${response.status}`);
    
    res.status(response.status).json(data);
  } catch (error) {
    console.error('[Proxy] Error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`[Proxy] CORS proxy server running on http://localhost:${PORT}`);
  console.log(`[Proxy] Forwarding requests to ${API_GATEWAY_URL}`);
  console.log(`[Proxy] Update your .env file:`);
  console.log(`[Proxy]   VITE_API_GATEWAY_URL=http://localhost:${PORT}`);
});
