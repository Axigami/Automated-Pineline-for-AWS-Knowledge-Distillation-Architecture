import type { EdgeNodeRow, EdgeNodeUIModel, NodeStatus } from './types';
import { normalizeEdgePresence } from '../../../utils/edgeNodeStatus';

function relativeTime(iso: string | null): string {
  if (!iso) return 'Chưa kết nối';
  const secs = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (secs < 60) return `${secs}s trước`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m trước`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h trước`;
  return `${Math.floor(secs / 86400)}d trước`;
}

export function adaptNode(row: EdgeNodeRow): EdgeNodeUIModel {
  return {
    id: row.id,
    nodeCode: row.node_code,
    location: row.location_text ?? '—',
    ipAddress: row.ip_address ?? '—',
    status: normalizeEdgePresence(row.status) as NodeStatus,
    modelVersion: row.model_version_text ?? '—',
    framework: row.framework ?? 'ONNX Runtime',
    lastSeenLabel: relativeTime(row.last_seen_at),
    lastSeenAt: row.last_seen_at,
    cpuPct: row.current_cpu_percent != null ? `${row.current_cpu_percent.toFixed(1)}%` : '—',
    cpuRaw: row.current_cpu_percent,
    memPct: row.current_ram_percent != null ? `${row.current_ram_percent.toFixed(1)}%` : '—',
    memRaw: row.current_ram_percent,
    tempC: row.current_temperature_c != null ? `${row.current_temperature_c.toFixed(1)}°C` : '—',
    tempRaw: row.current_temperature_c,
    latencyMs: row.current_latency_ms != null ? `${row.current_latency_ms}ms` : '—',
    latencyRaw: row.current_latency_ms,
  };
}

export function adaptNodes(rows: EdgeNodeRow[]): EdgeNodeUIModel[] {
  return rows.map(adaptNode);
}
