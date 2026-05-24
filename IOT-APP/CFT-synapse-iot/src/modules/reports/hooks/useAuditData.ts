import { useEffect, useState } from 'react';
import { supabase } from '../../../core/lib/supabaseClient';

export interface AuditRow {
  timestamp: string;
  user: string;
  action: string;
  target: string;
  status: string;
  source: 'alert' | 'retrain' | 'deployment';
}

export interface HomeOption {
  id: string;
  code: string;
  name: string;
}

export interface AuditData {
  auditRows: AuditRow[];
  homes: HomeOption[];
  total: number;
  loading: boolean;
  error: string | null;
}

export function useAuditData(): AuditData {
  const [data, setData] = useState<AuditData>({
    auditRows: [],
    homes: [],
    total: 0,
    loading: true,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    async function fetch() {
      try {
        const [alertsRes, retrainRes, deployRes, homesRes] = await Promise.all([
          supabase
            .from('alerts_all')
            .select('audit_created_at, audit_user_display_name, audit_action, audit_target, audit_status, alert_id')
            .not('audit_action', 'is', null)
            .order('audit_created_at', { ascending: false })
            .limit(20),

          supabase
            .from('retrain_jobs_all')
            .select('audit_created_at, audit_user_display_name, audit_action, audit_target, audit_status, job_id')
            .not('audit_action', 'is', null)
            .order('audit_created_at', { ascending: false })
            .limit(15),

          supabase
            .from('deployments_all')
            .select('audit_created_at, audit_user_display_name, audit_action, audit_target, audit_status, deployment_id')
            .not('audit_action', 'is', null)
            .order('audit_created_at', { ascending: false })
            .limit(15),

          supabase
            .from('homes')
            .select('id, code, name')
            .order('name', { ascending: true }),
        ]);

        if (alertsRes.error) throw new Error(alertsRes.error.message);
        if (retrainRes.error) throw new Error(retrainRes.error.message);
        if (deployRes.error) throw new Error(deployRes.error.message);
        if (homesRes.error) throw new Error(homesRes.error.message);
        if (cancelled) return;

        const alertRows: AuditRow[] = (alertsRes.data ?? []).map((r) => ({
          timestamp: r.audit_created_at ? new Date(r.audit_created_at).toLocaleString('sv-SE').slice(0, 19) : '—',
          user: r.audit_user_display_name ?? 'system',
          action: r.audit_action ?? '—',
          target: r.audit_target ?? r.alert_id?.slice(0, 8) ?? '—',
          status: r.audit_status ?? 'Success',
          source: 'alert' as const,
        }));

        const retrainRows: AuditRow[] = (retrainRes.data ?? []).map((r) => ({
          timestamp: r.audit_created_at ? new Date(r.audit_created_at).toLocaleString('sv-SE').slice(0, 19) : '—',
          user: r.audit_user_display_name ?? 'system',
          action: r.audit_action ?? '—',
          target: r.audit_target ?? r.job_id?.slice(0, 8) ?? '—',
          status: r.audit_status ?? 'Success',
          source: 'retrain' as const,
        }));

        const deployRows: AuditRow[] = (deployRes.data ?? []).map((r) => ({
          timestamp: r.audit_created_at ? new Date(r.audit_created_at).toLocaleString('sv-SE').slice(0, 19) : '—',
          user: r.audit_user_display_name ?? 'system',
          action: r.audit_action ?? '—',
          target: r.audit_target ?? r.deployment_id?.slice(0, 8) ?? '—',
          status: r.audit_status ?? 'Success',
          source: 'deployment' as const,
        }));

        const all = [...alertRows, ...retrainRows, ...deployRows].sort((a, b) =>
          b.timestamp.localeCompare(a.timestamp)
        );

        const homes: HomeOption[] = (homesRes.data ?? []).map((h) => ({
          id: h.id,
          code: h.code,
          name: h.name,
        }));

        if (!cancelled) {
          setData({ auditRows: all.slice(0, 50), homes, total: all.length, loading: false, error: null });
        }
      } catch (e: unknown) {
        if (!cancelled) {
          const msg = e instanceof Error ? e.message : String(e);
          setData((prev) => ({ ...prev, loading: false, error: msg }));
        }
      }
    }
    fetch();
    return () => { cancelled = true; };
  }, []);

  return data;
}
