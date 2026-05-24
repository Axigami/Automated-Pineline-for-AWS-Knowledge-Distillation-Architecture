import { useEffect, useCallback } from 'react';
import { supabase } from '../../../core/lib/supabaseClient';
import { useReportsStore } from '../model/store';

/**
 * useReports – Queries alerts_all audit columns for audit logs.
 * alert_created_at IS source for audit events.
 */
export function useReports() {
  const store = useReportsStore();

  const fetchData = useCallback(async () => {
    store.setIsLoading(true);

    const { data, error } = await supabase
      .from('alerts_all')
      .select(
        'alert_id, audit_user_id, audit_user_email, audit_user_display_name, audit_action, audit_target, audit_status, audit_created_at'
      )
      .not('audit_action', 'is', null)
      .order('audit_created_at', { ascending: false })
      .limit(200);

    if (error) {
      store.setError('Không thể tải audit logs: ' + error.message);
    } else {
      const logs = (data ?? []).map((row) => ({
        id: row.alert_id,
        timestamp: row.audit_created_at
          ? new Date(row.audit_created_at).toLocaleString('vi-VN')
          : '—',
        userId: row.audit_user_id ?? '—',
        username: row.audit_user_display_name ?? row.audit_user_email ?? '—',
        action: row.audit_action ?? '—',
        resource: row.audit_target ?? '—',
        status: row.audit_status ?? '—',
        ipAddress: '—', // không có trong schema
      }));
      store.setAuditLogs(logs);
    }

    store.setIsLoading(false);
  }, [store]);

  useEffect(() => { fetchData(); }, [fetchData]);

  return { ...store, refresh: fetchData };
}
