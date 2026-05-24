import { useEffect, useState } from 'react';
import { supabase } from './lib/supabaseClient';

export interface DbStatus {
  connected: boolean;
  loading: boolean;
  error: string | null;
}

export function useDbStatus(): DbStatus {
  const [status, setStatus] = useState<DbStatus>({ connected: false, loading: true, error: null });

  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const { error } = await supabase.from('homes').select('id', { count: 'exact', head: true });
        if (cancelled) return;
        if (error) {
          console.warn('[Supabase] Connection failed ❌', error.message);
          setStatus({ connected: false, loading: false, error: error.message });
        } else {
          console.log('[Supabase] Connected ✅');
          setStatus({ connected: true, loading: false, error: null });
        }
      } catch (e: unknown) {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : String(e);
        console.warn('[Supabase] Connection error ❌', msg);
        setStatus({ connected: false, loading: false, error: msg });
      }
    }
    check();
    return () => { cancelled = true; };
  }, []);

  return status;
}
