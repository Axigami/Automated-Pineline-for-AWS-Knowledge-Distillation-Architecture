import type {
  FlowRow,
  FlowUIModel,
  LabelAggregation,
  TopAttacker,
  TimelinePoint,
} from './types';

// Bảng màu cho từng nhãn tấn công (đồng bộ với UI)
const LABEL_COLORS: Record<string, string> = {
  PortScan: '#f43f5e',
  DDoS: '#3b82f6',
  Botnet: '#fbbf24',
  DoS: '#a855f7',
  BruteForce: '#f97316',
  'Web attack': '#06b6d4',
  'Web Attack': '#06b6d4',
  Benign: '#22c55e',
};
const DEFAULT_COLOR = '#64748b';

function formatBytes(bytes: number | null): string {
  if (bytes == null) return '—';
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(1)}MB`;
  if (bytes >= 1_024) return `${(bytes / 1024).toFixed(1)}K`;
  return `${bytes}`;
}

function formatDuration(sec: number | null): string {
  if (sec == null) return '—';
  return sec < 1 ? `${(sec * 1000).toFixed(0)}ms` : `${sec.toFixed(3)}s`;
}

export function adaptFlow(row: FlowRow): FlowUIModel {
  return {
    id: row.flow_id,
    homeId: row.flow_home_id ?? '',
    time: new Date(row.flow_ts).toLocaleString('vi-VN'),
    rawTs: row.flow_ts,
    srcIp: row.flow_src_ip ?? '—',
    dstIp: row.flow_dst_ip ?? '—',
    srcPort: row.flow_src_port != null ? String(row.flow_src_port) : '—',
    dstPort: row.flow_dst_port != null ? String(row.flow_dst_port) : '—',
    protocol: row.flow_protocol ?? '—',
    duration: formatDuration(row.flow_duration_s),
    inBytes: formatBytes(row.flow_in_bytes),
    outBytes: formatBytes(row.flow_out_bytes),
    tcpFlags: row.flow_tcp_flags ?? '—',
    predictedLabel: row.predicted_label === 'Web attack' ? 'Web Attack' : (row.predicted_label ?? 'Unknown'),
    trueLabel: row.feedback_true_label === 'Web attack' ? 'Web Attack' : (row.feedback_true_label ?? null),
    confidencePct:
      row.confidence != null ? `${Math.round(row.confidence * 100)}%` : '—',
    isAnomaly: row.is_anomaly ?? false,
    hasFeedback: row.feedback_action != null,
    inferenceLogic: row.inference_logic ?? '—',
  };
}

export function adaptFlows(rows: FlowRow[]): FlowUIModel[] {
  return rows.map(adaptFlow);
}

/** Tính Attack Distribution (PieChart) từ mảng FlowUIModel */
export function buildAggregation(flows: FlowUIModel[]): LabelAggregation[] {
  const countMap: Record<string, number> = {};
  flows.forEach((f) => {
    countMap[f.predictedLabel] = (countMap[f.predictedLabel] ?? 0) + 1;
  });
  return Object.entries(countMap)
    .sort((a, b) => b[1] - a[1])
    .map(([label, count]) => ({
      label,
      count,
      color: LABEL_COLORS[label] ?? DEFAULT_COLOR,
    }));
}

/** Tính Top Attacker IPs từ mảng FlowUIModel */
export function buildTopAttackers(flows: FlowUIModel[]): TopAttacker[] {
  const ipMap: Record<string, { count: number; label: string }> = {};
  flows.forEach((f) => {
    if (f.srcIp === '—') return;
    if (!ipMap[f.srcIp]) ipMap[f.srcIp] = { count: 0, label: f.predictedLabel };
    ipMap[f.srcIp].count += 1;
  });
  return Object.entries(ipMap)
    .sort((a, b) => b[1].count - a[1].count)
    .map(([ip, { count, label }]) => ({ ip, count, label }));
}

/** Tính Timeline (StackedBarChart) theo giờ từ mảng FlowUIModel */
export function buildTimeline(flows: FlowUIModel[]): TimelinePoint[] {
  // Pre-fill 24 buckets cho 24 giờ trong ngày để biểu đồ luôn cân xứng
  const buckets: Record<string, Record<string, number>> = {};
  for (let i = 0; i < 24; i++) {
    // format "0:00", "1:00", ... "23:00"
    buckets[`${i}:00`] = {};
  }

  flows.forEach((f) => {
    // Sử dụng rawTs thay vì slice/split chuỗi locale bị lỗi NaN
    const d = new Date(f.rawTs);
    if (isNaN(d.getTime())) return;

    // Gom theo giờ
    const hPart = `${d.getHours()}:00`;
    if (!buckets[hPart]) buckets[hPart] = {};
    const lbl = f.predictedLabel;
    buckets[hPart][lbl] = (buckets[hPart][lbl] ?? 0) + 1;
  });

  // Sort theo thứ tự giờ trong ngày
  return Object.entries(buckets)
    .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
    .map(([time, counts]) => ({ time, ...counts } as TimelinePoint));
}
