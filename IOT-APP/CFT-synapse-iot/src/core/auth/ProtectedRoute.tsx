import React from 'react';
import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { useAuthContext } from './AuthProvider';

interface ProtectedRouteProps {
  allowedRoles?: ('admin' | 'operator' | 'viewer')[];
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ allowedRoles }) => {
  const { session, role, loading } = useAuthContext();
  const location = useLocation();

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-slate-950">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-slate-400 font-medium">Verifying Session...</p>
        </div>
      </div>
    );
  }

  if (!session) {
    // Redirect to login page, save the location they tried to access
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  if (allowedRoles && allowedRoles.length > 0) {
    if (!role || !allowedRoles.includes(role)) {
      // User is logged in but doesn't have the required role
      // Redirect to a default dashboard or show an unauthorized page
      return (
        <div className="flex flex-col items-center justify-center h-screen bg-slate-950 text-slate-200">
          <div className="text-rose-500 mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>
          </div>
          <h1 className="text-2xl font-bold mb-2">Access Denied</h1>
          <p className="text-slate-500">You do not have permission to view this page.</p>
        </div>
      );
    }
  }

  // Render child routes
  return <Outlet />;
};
