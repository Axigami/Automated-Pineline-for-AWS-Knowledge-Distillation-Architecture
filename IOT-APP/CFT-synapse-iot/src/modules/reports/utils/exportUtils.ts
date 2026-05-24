import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import * as XLSX from 'xlsx';
import type {
  AlertRow,
  NetworkFlowRow,
  RetrainJobRow,
  ModelVersionRow,
  FleetHealthRow,
  AuditLogEntry,
  ReportSummaryStats,
  ReportFilters,
} from '../model/types';

export interface ExportPayload {
  filters: ReportFilters;
  summaryStats: ReportSummaryStats;
  alerts: AlertRow[];
  networkFlows: NetworkFlowRow[];
  retrainJobs: RetrainJobRow[];
  modelVersions: ModelVersionRow[];
  fleetHealth: FleetHealthRow[];
  auditLogs: AuditLogEntry[];
}

function fmtDate(ts: string | null | undefined): string {
  if (!ts) return '—';
  return new Date(ts).toLocaleString('en-US', { dateStyle: 'short', timeStyle: 'short' });
}

function fmtPct(v: number | null | undefined, isRaw = true): string {
  if (v == null) return '—';
  return isRaw ? `${(v * 100).toFixed(1)}%` : `${v.toFixed(1)}%`;
}

// ── PDF Export ───────────────────────────────────────────────────────────────
export async function exportToPDF(
  elementId: string,
  filename = 'report',
): Promise<void> {
  const el = document.getElementById(elementId);
  if (!el) throw new Error(`Element #${elementId} not found`);

  const canvas = await html2canvas(el, {
    scale: 2,
    useCORS: true,
    backgroundColor: '#0f172a',
    logging: false,
  });

  const imgData = canvas.toDataURL('image/png');
  const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
  const pdfW = pdf.internal.pageSize.getWidth();
  const pdfH = pdf.internal.pageSize.getHeight();
  const imgH = (canvas.height * pdfW) / canvas.width;

  let position = 0;
  let remaining = imgH;
  while (remaining > 0) {
    pdf.setFillColor('#020617');
    pdf.rect(0, 0, pdfW, pdfH, 'F');
    pdf.addImage(imgData, 'PNG', 0, position, pdfW, imgH);
    remaining -= pdfH;
    if (remaining > 0) { pdf.addPage(); position -= pdfH; }
  }
  pdf.save(`${filename}.pdf`);
}

// ── Excel Export  (multi-sheet) ───────────────────────────────────────────────
export function exportToExcel(payload: ExportPayload, filename = 'report'): void {
  const wb = XLSX.utils.book_new();
  const { filters } = payload;
  const period = filters.dateRange.from && filters.dateRange.to
    ? `${filters.dateRange.from} → ${filters.dateRange.to}`
    : 'All time';

  // Sheet 1 — Summary
  const summarySheet = XLSX.utils.aoa_to_sheet([
    ['SYNAPSE IOT — SECURITY REPORT'],
    [`Period: ${period}`],
    [`Generated: ${new Date().toLocaleString('en-US')}`],
    [],
    ['METRIC', 'VALUE'],
    ['Total Alerts', payload.summaryStats.totalAlerts],
    ['Critical / High Alerts', payload.summaryStats.criticalAlerts],
    ['Resolved Alerts', payload.summaryStats.resolvedAlerts],
    ['Total Network Flows', payload.summaryStats.totalFlows],
    ['Anomalous Flows', payload.summaryStats.anomalyFlows],
    ['Retrain Jobs', payload.summaryStats.retrainJobs],
    ['Avg Model Accuracy', `${payload.summaryStats.avgModelAccuracy}%`],
    ['Online Nodes', `${payload.summaryStats.onlineNodes} / ${payload.summaryStats.totalNodes}`],
  ]);
  XLSX.utils.book_append_sheet(wb, summarySheet, 'Summary');

  // Sheet 2 — Security Alerts
  if (filters.sections.attackSummary && payload.alerts.length) {
    XLSX.utils.book_append_sheet(
      wb,
      XLSX.utils.aoa_to_sheet([
        ['Alert ID', 'First Seen', 'Threat Type', 'Severity', 'Status',
          'Source IP', 'Destination IP', 'Confidence', 'Predicted Label', 'Home', 'Node'],
        ...payload.alerts.map((a) => [
          a.alert_id.slice(0, 8),
          fmtDate(a.alert_first_seen_at),
          a.alert_threat_type,
          a.alert_severity,
          a.alert_status,
          a.alert_source_ip ?? '—',
          a.alert_target_ip ?? '—',
          fmtPct(a.alert_confidence),
          a.alert_predicted_label ?? '—',
          a.home_name ?? '—',
          a.node_code ?? '—',
        ]),
      ]),
      'Security Alerts',
    );
  }

  // Sheet 3 — Network Flows
  if (filters.sections.networkFlows && payload.networkFlows.length) {
    XLSX.utils.book_append_sheet(
      wb,
      XLSX.utils.aoa_to_sheet([
        ['Flow ID', 'Timestamp', 'Protocol', 'Src IP', 'Src Port',
          'Dst IP', 'Dst Port', 'Total Bytes', 'Duration (s)',
          'Anomaly', 'Predicted Label', 'Confidence', 'Score'],
        ...payload.networkFlows.map((f) => [
          f.flow_id.slice(0, 8),
          fmtDate(f.flow_ts),
          f.flow_protocol ?? '—',
          f.flow_src_ip ?? '—',
          f.flow_src_port ?? '—',
          f.flow_dst_ip ?? '—',
          f.flow_dst_port ?? '—',
          f.flow_total_bytes ?? 0,
          f.flow_duration_s ?? '—',
          f.is_anomaly ? 'YES' : 'NO',
          f.predicted_label ?? '—',
          fmtPct(f.confidence),
          f.anomaly_score != null ? f.anomaly_score.toFixed(4) : '—',
        ]),
      ]),
      'Network Flows',
    );
  }

  // Sheet 4 — Model Versions
  if (filters.sections.modelAccuracy && payload.modelVersions.length) {
    XLSX.utils.book_append_sheet(
      wb,
      XLSX.utils.aoa_to_sheet([
        ['Model', 'Version', 'Status', 'Accuracy', 'F1-Score',
          'Precision', 'Recall', 'FPR', 'Latency (ms)', 'Memory (MB)', 'Throughput/s', 'Author', 'Created'],
        ...payload.modelVersions.map((m) => [
          m.model_name ?? '—',
          m.version,
          m.status,
          fmtPct(m.accuracy),
          m.f1_score != null ? m.f1_score.toFixed(4) : '—',
          m.precision != null ? m.precision.toFixed(4) : '—',
          m.recall != null ? m.recall.toFixed(4) : '—',
          m.false_positive_rate != null ? m.false_positive_rate.toFixed(4) : '—',
          m.latency_ms ?? '—',
          m.memory_mb ?? '—',
          m.throughput_per_s != null ? m.throughput_per_s.toFixed(1) : '—',
          m.author ?? '—',
          fmtDate(m.created_at),
        ]),
      ]),
      'Model Versions',
    );
  }

  // Sheet 5 — Retrain Jobs
  if (filters.sections.retrainJobs && payload.retrainJobs.length) {
    XLSX.utils.book_append_sheet(
      wb,
      XLSX.utils.aoa_to_sheet([
        ['Job ID', 'Status', 'Created', 'Started', 'Finished',
          'Epochs', 'Progress', 'KD', 'Data Range', 'Performed By', 'Audit Status', 'Home'],
        ...payload.retrainJobs.map((j) => [
          j.job_id.slice(0, 8),
          j.job_status,
          fmtDate(j.job_created_at),
          fmtDate(j.job_started_at),
          fmtDate(j.job_finished_at),
          j.job_epochs ?? '—',
          j.job_progress_percent != null ? `${j.job_progress_percent}%` : '—',
          j.job_knowledge_distillation ? 'Yes' : 'No',
          j.job_data_range ?? '—',
          j.audit_user_display_name ?? '—',
          j.audit_status ?? '—',
          j.home_name ?? '—',
        ]),
      ]),
      'Retrain Jobs',
    );
  }

  // Sheet 6 — Fleet Health
  if (filters.sections.fleetHealth && payload.fleetHealth.length) {
    XLSX.utils.book_append_sheet(
      wb,
      XLSX.utils.aoa_to_sheet([
        ['Node Code', 'Status', 'IP Address', 'CPU %', 'RAM %',
          'Temp °C', 'Latency ms', 'Last Seen', 'Framework', 'Model Ver.', 'Home'],
        ...payload.fleetHealth.map((n) => [
          n.node_code,
          n.status,
          n.ip_address ?? '—',
          n.current_cpu_percent != null ? n.current_cpu_percent.toFixed(1) : '—',
          n.current_ram_percent != null ? n.current_ram_percent.toFixed(1) : '—',
          n.current_temperature_c != null ? n.current_temperature_c.toFixed(1) : '—',
          n.current_latency_ms ?? '—',
          fmtDate(n.last_seen_at),
          n.framework ?? '—',
          n.model_version_text ?? '—',
          n.home_name ?? '—',
        ]),
      ]),
      'Fleet Health',
    );
  }

  // Sheet 7 — Audit Trail
  if (filters.sections.auditTrail && payload.auditLogs.length) {
    XLSX.utils.book_append_sheet(
      wb,
      XLSX.utils.aoa_to_sheet([
        ['Timestamp', 'Username', 'Email', 'Action', 'Resource', 'Status', 'Source'],
        ...payload.auditLogs.map((l) => [
          l.timestamp, l.username, l.user_email,
          l.action, l.resource, l.status, l.source,
        ]),
      ]),
      'Audit Trail',
    );
  }

  XLSX.writeFile(wb, `${filename}.xlsx`);
}

// ── CSV Export (Audit Trail only) ─────────────────────────────────────────────
export function exportAuditToCSV(auditLogs: AuditLogEntry[], filename = 'audit_trail'): void {
  const ws = XLSX.utils.aoa_to_sheet([
    ['Timestamp', 'Username', 'Email', 'Action', 'Resource', 'Status', 'Source'],
    ...auditLogs.map((l) => [l.timestamp, l.username, l.user_email, l.action, l.resource, l.status, l.source]),
  ]);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, 'Audit Trail');
  XLSX.writeFile(wb, `${filename}.csv`, { bookType: 'csv' });
}
