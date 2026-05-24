/**
 * Fleet Management – Types aligned to edge_nodes + node_telemetry schema
 */
/** Chỉ hai trạng thái hiển thị (xanh / đỏ). Giá trị DB khác được map trong adapter. */
export type NodeStatus = 'online' | 'offline';

// ─── DB Row Types ──────────────────────────────────────────────────────────────

export interface EdgeNodeRow {
  id: string;
  home_id: string;
  node_code: string;
  status: string;
  location_text: string | null;
  ip_address: string | null;
  last_seen_at: string | null;
  current_cpu_percent: number | null;
  current_ram_percent: number | null;
  current_temperature_c: number | null;
  current_latency_ms: number | null;
  model_version_text: string | null;
  framework: string | null;
  deployed_model_version_id: string | null;
}

export interface NodeTelemetryRow {
  id: string;
  node_id: string;
  ts: string;
  cpu_percent: number | null;
  ram_percent: number | null;
  temperature_c: number | null;
  latency_ms: number | null;
}

// ─── UI Model Types ───────────────────────────────────────────────────────────

export interface TelemetryPoint {
  ts: string;
  cpu: number;
  ram: number;
  temp: number;
  latency: number;
}

export interface EdgeNodeUIModel {
  id: string;
  nodeCode: string;
  location: string;
  ipAddress: string;
  status: NodeStatus;
  modelVersion: string;
  framework: string;
  lastSeenLabel: string;
  lastSeenAt: string | null;
  cpuPct: string;
  cpuRaw: number | null;
  memPct: string;
  memRaw: number | null;
  tempC: string;
  tempRaw: number | null;
  latencyMs: string;
  latencyRaw: number | null;
}
