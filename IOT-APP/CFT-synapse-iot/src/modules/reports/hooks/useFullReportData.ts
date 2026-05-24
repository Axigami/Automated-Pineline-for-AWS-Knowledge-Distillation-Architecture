import { useEffect, useState, useCallback } from 'react';
import { supabase } from '../../../core/lib/supabaseClient';
import type {
  AuditLogEntry,
  AlertRow,
  NetworkFlowRow,
  RetrainJobRow,
  ModelVersionRow,
  FleetHealthRow,
  ReportSummaryStats,
  ReportFilters,
  Home,
} from '../model/types';

// ─────────────────────────────────────────────────────────────────────────────
// Public shape returned by the hook
// ─────────────────────────────────────────────────────────────────────────────
export interface FullReportData {
  homes: Home[];
  summaryStats: ReportSummaryStats;
  auditLogs: AuditLogEntry[];
  alerts: AlertRow[];
  networkFlows: NetworkFlowRow[];
  retrainJobs: RetrainJobRow[];
  modelVersions: ModelVersionRow[];
  fleetHealth: FleetHealthRow[];
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

const EMPTY_STATS: ReportSummaryStats = {
  totalAlerts: 0, criticalAlerts: 0, resolvedAlerts: 0,
  totalFlows: 0, anomalyFlows: 0, retrainJobs: 0,
  avgModelAccuracy: 0, onlineNodes: 0, totalNodes: 0,
};

// ─────────────────────────────────────────────────────────────────────────────
// Hook
// ─────────────────────────────────────────────────────────────────────────────
export function useFullReportData(filters: ReportFilters): FullReportData {
  // Stable stringified key drives re-fetch
  const filterKey = `${filters.dateRange.from}|${filters.dateRange.to}|${filters.homeId}`;

  const [homes, setHomes] = useState<Home[]>([]);
  const [summaryStats, setSummaryStats] = useState<ReportSummaryStats>(EMPTY_STATS);
  const [auditLogs, setAuditLogs] = useState<AuditLogEntry[]>([]);
  const [alerts, setAlerts] = useState<AlertRow[]>([]);
  const [networkFlows, setNetworkFlows] = useState<NetworkFlowRow[]>([]);
  const [retrainJobs, setRetrainJobs] = useState<RetrainJobRow[]>([]);
  const [modelVersions, setModelVersions] = useState<ModelVersionRow[]>([]);
  const [fleetHealth, setFleetHealth] = useState<FleetHealthRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAll = useCallback(async () => {
    // Read current filter values inside callback to avoid stale closure
    const from   = filters.dateRange.from;
    const to     = filters.dateRange.to;
    const homeId = filters.homeId;
    const hasDates = Boolean(from && to);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const addDates = (q: any, col: string) =>
      hasDates ? q.gte(col, `${from}T00:00:00`).lte(col, `${to}T23:59:59`) : q;

    setLoading(true);
    setError(null);

    try {
      // ── 1. Homes ─────────────────────────────────────────────────────────
      const { data: homesRaw, error: eHomes } = await supabase
        .from('homes')
        .select('id, code, name, region')
        .order('name');
      if (eHomes) throw new Error(`homes: ${eHomes.message}`);

      const homesArr: Home[] = (homesRaw ?? []).map((r) => ({
        id: r.id, code: r.code, name: r.name, region: r.region ?? null,
      }));
      const homeMap: Record<string, string> = Object.fromEntries(
        homesArr.map((h) => [h.id, h.name]),
      );
      setHomes(homesArr);

      // ── 2. Models (for name lookup) ─────────────────────────────────────
      const { data: modelsRaw } = await supabase
        .from('models')
        .select('id, name');
      const modelMap: Record<string, string> = Object.fromEntries(
        (modelsRaw ?? []).map((m) => [m.id, m.name]),
      );

      // ── 3. Edge Nodes (Fleet Health) ────────────────────────────────────
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let enQ: any = supabase
        .from('edge_nodes')
        .select(
          'id, node_code, home_id, status, ip_address, last_seen_at,' +
          'current_cpu_percent, current_ram_percent, current_temperature_c,' +
          'current_latency_ms, framework, model_version_text',
        )
        .order('status');
      if (homeId) enQ = enQ.eq('home_id', homeId);

      const { data: nodesRaw, error: eNodes } = await enQ;
      if (eNodes) throw new Error(`edge_nodes: ${eNodes.message}`);

      const fleetArr: FleetHealthRow[] = (nodesRaw ?? []).map((r: Record<string, unknown>) => ({
        id: r.id as string,
        node_code: r.node_code as string,
        home_id: r.home_id as string,
        status: r.status as string,
        ip_address: r.ip_address as string | null,
        last_seen_at: r.last_seen_at as string | null,
        current_cpu_percent: r.current_cpu_percent as number | null,
        current_ram_percent: r.current_ram_percent as number | null,
        current_temperature_c: r.current_temperature_c as number | null,
        current_latency_ms: r.current_latency_ms as number | null,
        framework: r.framework as string | null,
        model_version_text: r.model_version_text as string | null,
        home_name: r.home_id ? homeMap[r.home_id as string] : undefined,
      }));
      setFleetHealth(fleetArr);

      // ── 4. Alerts ────────────────────────────────────────────────────────
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let alertQ: any = supabase
        .from('alerts_all')
        .select(
          'alert_id, alert_home_id, alert_node_id, alert_first_seen_at,' +
          'alert_threat_type, alert_severity, alert_status,' +
          'alert_confidence, alert_predicted_label,' +
          'alert_source_ip, alert_target_ip, alert_verdict_text, alert_verified_at',
        )
        .order('alert_first_seen_at', { ascending: false })
        .limit(500);
      alertQ = addDates(alertQ, 'alert_first_seen_at');
      if (homeId) alertQ = alertQ.eq('alert_home_id', homeId);

      const { data: alertsRaw, error: eAlerts } = await alertQ;
      if (eAlerts) throw new Error(`alerts_all: ${eAlerts.message}`);

      // Build node_code lookup from already-fetched nodes
      const nodeCodeMap: Record<string, string> = Object.fromEntries(
        fleetArr.map((n) => [n.id, n.node_code]),
      );

      const alertsArr: AlertRow[] = (alertsRaw ?? []).map((r: Record<string, unknown>) => ({
        alert_id: r.alert_id as string,
        alert_home_id: r.alert_home_id as string,
        alert_node_id: r.alert_node_id as string | null,
        alert_first_seen_at: r.alert_first_seen_at as string,
        alert_threat_type: r.alert_threat_type as string,
        alert_severity: r.alert_severity as string,
        alert_status: r.alert_status as string,
        alert_confidence: r.alert_confidence as number | null,
        alert_predicted_label: r.alert_predicted_label as string | null,
        alert_source_ip: r.alert_source_ip as string | null,
        alert_target_ip: r.alert_target_ip as string | null,
        alert_verdict_text: r.alert_verdict_text as string | null,
        alert_verified_at: r.alert_verified_at as string | null,
        home_name: r.alert_home_id ? homeMap[r.alert_home_id as string] : undefined,
        node_code: r.alert_node_id ? nodeCodeMap[r.alert_node_id as string] : undefined,
      }));
      setAlerts(alertsArr);

      // ── 5. Network Flows ─────────────────────────────────────────────────
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let flowQ: any = supabase
        .from('network_flows_feedback_all')
        .select(
          'flow_id, flow_home_id, flow_node_id, flow_ts, flow_protocol,' +
          'flow_src_ip, flow_dst_ip, flow_src_port, flow_dst_port,' +
          'flow_total_bytes, flow_duration_s,' +
          'is_anomaly, predicted_label, confidence, anomaly_score,' +
          'feedback_action, feedback_true_label',
        )
        .order('flow_ts', { ascending: false })
        .limit(300);
      flowQ = addDates(flowQ, 'flow_ts');
      if (homeId) flowQ = flowQ.eq('flow_home_id', homeId);

      const { data: flowsRaw, error: eFlows } = await flowQ;
      if (eFlows) throw new Error(`network_flows_feedback_all: ${eFlows.message}`);

      const flowsArr: NetworkFlowRow[] = (flowsRaw ?? []).map((r: Record<string, unknown>) => ({
        flow_id: r.flow_id as string,
        flow_home_id: r.flow_home_id as string,
        flow_node_id: r.flow_node_id as string | null,
        flow_ts: r.flow_ts as string,
        flow_protocol: r.flow_protocol as string | null,
        flow_src_ip: r.flow_src_ip as string | null,
        flow_dst_ip: r.flow_dst_ip as string | null,
        flow_src_port: r.flow_src_port as number | null,
        flow_dst_port: r.flow_dst_port as number | null,
        flow_total_bytes: r.flow_total_bytes as number | null,
        flow_duration_s: r.flow_duration_s as number | null,
        is_anomaly: r.is_anomaly as boolean | null,
        predicted_label: r.predicted_label as string | null,
        confidence: r.confidence as number | null,
        anomaly_score: r.anomaly_score as number | null,
        feedback_action: r.feedback_action as string | null,
        feedback_true_label: r.feedback_true_label as string | null,
        home_name: r.flow_home_id ? homeMap[r.flow_home_id as string] : undefined,
      }));
      setNetworkFlows(flowsArr);

      // ── 6. Retrain Jobs ──────────────────────────────────────────────────
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let rjQ: any = supabase
        .from('retrain_jobs_all')
        .select(
          'job_id, job_home_id, job_status, job_created_at,' +
          'job_started_at, job_finished_at, job_epochs,' +
          'job_progress_percent, job_knowledge_distillation, job_data_range,' +
          'audit_user_display_name, audit_action, audit_status',
        )
        .order('job_created_at', { ascending: false })
        .limit(100);
      rjQ = addDates(rjQ, 'job_created_at');
      if (homeId) rjQ = rjQ.eq('job_home_id', homeId);

      const { data: retrainRaw, error: eRetrain } = await rjQ;
      if (eRetrain) throw new Error(`retrain_jobs_all: ${eRetrain.message}`);

      const retrainArr: RetrainJobRow[] = (retrainRaw ?? []).map((r: Record<string, unknown>) => ({
        job_id: r.job_id as string,
        job_home_id: r.job_home_id as string | null,
        job_status: r.job_status as string,
        job_created_at: r.job_created_at as string,
        job_started_at: r.job_started_at as string | null,
        job_finished_at: r.job_finished_at as string | null,
        job_epochs: r.job_epochs as number | null,
        job_progress_percent: r.job_progress_percent as number | null,
        job_knowledge_distillation: r.job_knowledge_distillation as boolean | null,
        job_data_range: r.job_data_range as string | null,
        audit_user_display_name: r.audit_user_display_name as string | null,
        audit_action: r.audit_action as string | null,
        audit_status: r.audit_status as string | null,
        home_name: r.job_home_id ? homeMap[r.job_home_id as string] : undefined,
      }));
      setRetrainJobs(retrainArr);

      // ── 7. Model Versions ────────────────────────────────────────────────
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let mvQ: any = supabase
        .from('model_versions')
        .select(
          'id, model_id, version, status, created_at, author,' +
          'accuracy, f1_score, precision, recall, false_positive_rate,' +
          'latency_ms, memory_mb, throughput_per_s',
        )
        .order('created_at', { ascending: false })
        .limit(50);
      mvQ = addDates(mvQ, 'created_at');

      const { data: mvRaw, error: eMV } = await mvQ;
      if (eMV) throw new Error(`model_versions: ${eMV.message}`);

      const modelsArr: ModelVersionRow[] = (mvRaw ?? []).map((r: Record<string, unknown>) => ({
        id: r.id as string,
        model_id: r.model_id as string,
        version: r.version as string,
        status: r.status as string,
        created_at: r.created_at as string,
        author: r.author as string | null,
        accuracy: r.accuracy as number | null,
        f1_score: r.f1_score as number | null,
        precision: r.precision as number | null,
        recall: r.recall as number | null,
        false_positive_rate: r.false_positive_rate as number | null,
        latency_ms: r.latency_ms as number | null,
        memory_mb: r.memory_mb as number | null,
        throughput_per_s: r.throughput_per_s as number | null,
        model_name: r.model_id ? modelMap[r.model_id as string] : undefined,
      }));
      setModelVersions(modelsArr);

      // ── 8. Audit Trail ───────────────────────────────────────────────────
      // Collect from 3 tables in parallel, then merge + sort
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let aaQ: any = supabase
        .from('alerts_all')
        .select('alert_id, audit_user_display_name, audit_user_email, audit_action, audit_target, audit_status, audit_created_at')
        .not('audit_action', 'is', null)
        .order('audit_created_at', { ascending: false })
        .limit(200);
      aaQ = addDates(aaQ, 'audit_created_at');

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let rjAQ: any = supabase
        .from('retrain_jobs_all')
        .select('job_id, audit_user_display_name, audit_user_email, audit_action, audit_target, audit_status, audit_created_at')
        .not('audit_action', 'is', null)
        .order('audit_created_at', { ascending: false })
        .limit(100);
      rjAQ = addDates(rjAQ, 'audit_created_at');

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let dpQ: any = supabase
        .from('deployments_all')
        .select('deployment_id, audit_user_display_name, audit_user_email, audit_action, audit_target, audit_status, audit_created_at')
        .not('audit_action', 'is', null)
        .order('audit_created_at', { ascending: false })
        .limit(100);
      dpQ = addDates(dpQ, 'audit_created_at');

      const [{ data: aaData }, { data: rjAData }, { data: dpData }] = await Promise.all([aaQ, rjAQ, dpQ]);

      const toAudit = (
        r: Record<string, unknown>,
        idField: string,
        source: AuditLogEntry['source'],
      ): AuditLogEntry => ({
        id: String(r[idField] ?? ''),
        timestamp: r.audit_created_at
          ? new Date(r.audit_created_at as string).toISOString().slice(0, 19).replace('T', ' ')
          : '—',
        username: String(r.audit_user_display_name ?? 'system'),
        user_email: String(r.audit_user_email ?? ''),
        action: String(r.audit_action ?? '—'),
        resource: String(r.audit_target ?? '—'),
        status: String(r.audit_status ?? '—'),
        source,
      });

      const auditArr: AuditLogEntry[] = [
        ...(aaData ?? []).map((r: Record<string, unknown>) => toAudit(r, 'alert_id', 'alert')),
        ...(rjAData ?? []).map((r: Record<string, unknown>) => toAudit(r, 'job_id', 'retrain')),
        ...(dpData ?? []).map((r: Record<string, unknown>) => toAudit(r, 'deployment_id', 'deployment')),
      ].sort((a, b) => b.timestamp.localeCompare(a.timestamp));
      setAuditLogs(auditArr);

      // ── 9. Summary stats ─────────────────────────────────────────────────
      const validAcc = modelsArr.filter((m) => m.accuracy != null);
      const avgAcc =
        validAcc.length > 0
          ? Math.round((validAcc.reduce((s, m) => s + (m.accuracy ?? 0), 0) / validAcc.length) * 10000) / 100
          : 0;

      setSummaryStats({
        totalAlerts: alertsArr.length,
        criticalAlerts: alertsArr.filter((a) => a.alert_severity === 'critical' || a.alert_severity === 'high').length,
        resolvedAlerts: alertsArr.filter((a) => a.alert_status === 'resolved' || a.alert_status === 'closed').length,
        totalFlows: flowsArr.length,
        anomalyFlows: flowsArr.filter((f) => f.is_anomaly === true).length,
        retrainJobs: retrainArr.length,
        avgModelAccuracy: avgAcc,
        onlineNodes: fleetArr.filter((n) => n.status === 'online').length,
        totalNodes: fleetArr.length,
      });

      setLoading(false);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error('[useFullReportData]', msg);
      setError(msg);
      setLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filterKey]);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  return {
    homes, summaryStats, auditLogs, alerts,
    networkFlows, retrainJobs, modelVersions, fleetHealth,
    loading, error, refetch: fetchAll,
  };
}
