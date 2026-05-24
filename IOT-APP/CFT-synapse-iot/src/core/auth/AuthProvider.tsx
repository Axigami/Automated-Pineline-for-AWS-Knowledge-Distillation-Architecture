import React, { createContext, useContext, useEffect, useState } from 'react';
import { Session, User } from '@supabase/supabase-js';
import { supabase } from '../lib/supabaseClient';
import { clearAllDashboardCaches } from '../../modules/dashboard/model/cache';

export type Rolecode = 'admin' | 'operator' | 'viewer' | 'soc level 3' | null;

interface AuthContextType {
  session: Session | null;
  user: User | null;
  role: Rolecode;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType>({
  session: null,
  user: null,
  role: null,
  loading: true,
});

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [session, setSession] = useState<Session | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [role, setRole] = useState<Rolecode>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    async function fetchRole(userId: string) {
      try {
        const { data, error } = await supabase
          .from('users_roles_settings')
          .select('role_code')
          .eq('user_id', userId)
          .limit(1)
          .maybeSingle();

        if (error) {
          console.warn('Failed to fetch user role:', error.message);
          return null;
        }
        const rowData = data as any;
        return (rowData && rowData.role_code) ? rowData.role_code.toLowerCase() as Rolecode : null;
      } catch (err) {
        console.error('Error fetching role:', err);
        return null;
      }
    }

    async function initializeAuth() {
      const { data: { session }, error } = await supabase.auth.getSession();

      if (error) {
        console.error('Error getting session:', error.message);
      }

      if (!mounted) return;

      setSession(session);
      setUser(session?.user ?? null);
      setRole(null);
      setLoading(false);

      if (session?.user) {
        void fetchRole(session.user.id).then((userRole) => {
          if (mounted) setRole(userRole);
        });
      }
    }

    initializeAuth();

    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, currentSession) => {
      if (!mounted) return;

      if (event === 'SIGNED_OUT') {
        clearAllDashboardCaches();
      }

      setSession(currentSession);
      setUser(currentSession?.user ?? null);
      setLoading(false);

      if (!currentSession?.user) {
        setRole(null);
        return;
      }

      if (event === 'SIGNED_IN' || event === 'INITIAL_SESSION') {
        void fetchRole(currentSession.user.id).then((userRole) => {
          if (mounted) setRole(userRole);
        });
      }
    });

    return () => {
      mounted = false;
      subscription.unsubscribe();
    };
  }, []);

  return (
    <AuthContext.Provider value={{ session, user, role, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuthContext = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuthContext must be used within an AuthProvider');
  }
  return context;
};
