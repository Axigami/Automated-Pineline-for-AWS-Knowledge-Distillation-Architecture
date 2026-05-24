import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { AuthProvider } from './core/auth/AuthProvider';
import App from './App.tsx';
import './index.css';
import 'leaflet/dist/leaflet.css';

// StrictMode đã được tắt để tránh Supabase auth-token lock warning
// do double-mount trong development mode
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <AuthProvider>
        <App />
      </AuthProvider>
    </BrowserRouter>
  </StrictMode>,
);
