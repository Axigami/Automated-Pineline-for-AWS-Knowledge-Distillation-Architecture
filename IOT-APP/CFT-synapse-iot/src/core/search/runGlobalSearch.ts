import { supabase } from '../lib/supabaseClient';

export type GlobalSearchNode = {
  type: 'node';
  id: string;
  node_code: string;
  ip_address: string | null;
  location_text: string | null;
  status: string | null;
};

export type GlobalSearchAlert = {
  type: 'alert';
  alert_id: string;
  alert_threat_type: string;
  alert_source_ip: string | null;
  alert_severity: string;
  alert_created_at: string;
};

function ilikePattern(raw: string): string | null {
  const t = raw.trim();
  if (t.length < 2) return null;
  const escaped = t.replace(/\\/g, '\\\\').replace(/%/g, '\\%').replace(/_/g, '\\_');
  return `%${escaped}%`;
}

function mapNode(row: any): GlobalSearchNode {
  return {
    type: 'node',
    id: row.id,
    node_code: row.node_code,
    ip_address: row.ip_address,
    location_text: row.location_text,
    status: row.status,
  };
}

function mapAlert(row: any): GlobalSearchAlert {
  return {
    type: 'alert',
    alert_id: row.alert_id,
    alert_threat_type: row.alert_threat_type,
    alert_source_ip: row.alert_source_ip,
    alert_severity: row.alert_severity,
    alert_created_at: row.alert_created_at,
  };
}

export async function runGlobalSearch(q: string): Promise<{
  nodes: GlobalSearchNode[];
  alerts: GlobalSearchAlert[];
}> {
  const p = ilikePattern(q);
  if (!p) return { nodes: [], alerts: [] };

  const nodeSelect = 'id, node_code, ip_address, location_text, status';
  const alertSelect = 'alert_id, alert_threat_type, alert_source_ip, alert_severity, alert_created_at';

  const [nc, nip, nloc, at, asip, atip, albl] = await Promise.all([
    supabase.from('edge_nodes' as any).select(nodeSelect).ilike('node_code', p).limit(6),
    supabase.from('edge_nodes' as any).select(nodeSelect).ilike('ip_address', p).limit(6),
    supabase.from('edge_nodes' as any).select(nodeSelect).ilike('location_text', p).limit(6),
    supabase.from('alerts_all' as any).select(alertSelect).ilike('alert_threat_type', p).order('alert_created_at', { ascending: false }).limit(6),
    supabase.from('alerts_all' as any).select(alertSelect).ilike('alert_source_ip', p).order('alert_created_at', { ascending: false }).limit(6),
    supabase.from('alerts_all' as any).select(alertSelect).ilike('alert_target_ip', p).order('alert_created_at', { ascending: false }).limit(6),
    supabase.from('alerts_all' as any).select(alertSelect).ilike('alert_predicted_label', p).order('alert_created_at', { ascending: false }).limit(6),
  ]);

  [nc, nip, nloc, at, asip, atip, albl].forEach((r, i) => {
    if (r.error) console.warn(`Global search query ${i}:`, r.error.message);
  });

  const nodeMap = new Map<string, any>();
  for (const row of [...(nc.data ?? []), ...(nip.data ?? []), ...(nloc.data ?? [])]) {
    nodeMap.set(row.id, row);
  }
  const nodes = Array.from(nodeMap.values()).slice(0, 8).map(mapNode);

  const alertMap = new Map<string, any>();
  for (const row of [...(at.data ?? []), ...(asip.data ?? []), ...(atip.data ?? []), ...(albl.data ?? [])]) {
    alertMap.set(row.alert_id, row);
  }
  const alerts = Array.from(alertMap.values())
    .sort((a, b) => new Date(b.alert_created_at).getTime() - new Date(a.alert_created_at).getTime())
    .slice(0, 8)
    .map(mapAlert);

  return { nodes, alerts };
}
