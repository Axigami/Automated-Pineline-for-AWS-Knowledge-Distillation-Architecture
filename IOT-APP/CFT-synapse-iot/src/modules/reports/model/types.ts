// ─────────────────────────────────────────────────────────────────────────────
// Reports Module — Type Definitions
// Aligned with Supabase schema (Database_App_IoT.sql)
// ─────────────────────────────────────────────────────────────────────────────

export interface Home {
  id: string;
  code: string;
  name: string;
  region: string | null;
}

/** Synthesised audit-trail row (sourced from alerts_all, retrain_jobs_all, deployments_all) */
export interface AuditLogEntry {
  id: string;
  timestamp: string;        // ISO display string
  username: string;
  user_email: string;
  action: string;
  resource: string;
  status: string;
  source: 'alert' | 'retrain' | 'deployment';
}

/** Source: alerts_all */
export interface AlertRow {
  alert_id: string;
  alert_home_id: string;
  alert_node_id: string | null;
  alert_first_seen_at: string;
  alert_threat_type: string;
  alert_severity: string;           // critical | high | medium | low
  alert_status: string;             // open | investigating | resolved | closed
  alert_confidence: number | null;  // 0–1 float
  alert_predicted_label: string | null;
  alert_source_ip: string | null;
  alert_target_ip: string | null;
  alert_verdict_text: string | null;
  alert_verified_at: string | null;
  // client-side joins
  home_name?: string;
  node_code?: string;
}

/** Source: network_flows_feedback_all */
export interface NetworkFlowRow {
  flow_id: string;
  flow_home_id: string;
  flow_node_id: string | null;
  flow_ts: string;
  flow_protocol: string | null;
  flow_src_ip: string | null;
  flow_dst_ip: string | null;
  flow_src_port: number | null;
  flow_dst_port: number | null;
  flow_total_bytes: number | null;
  flow_duration_s: number | null;
  is_anomaly: boolean | null;
  predicted_label: string | null;
  confidence: number | null;        // 0–1 float
  anomaly_score: number | null;
  feedback_action: string | null;
  feedback_true_label: string | null;
  // client-side joins
  home_name?: string;
}

/** Source: retrain_jobs_all */
export interface RetrainJobRow {
  job_id: string;
  job_home_id: string | null;
  job_status: string;               // pending | running | completed | failed
  job_created_at: string;
  job_started_at: string | null;
  job_finished_at: string | null;
  job_epochs: number | null;
  job_progress_percent: number | null;
  job_knowledge_distillation: boolean | null;
  job_data_range: string | null;
  audit_user_display_name: string | null;
  audit_action: string | null;
  audit_status: string | null;
  // client-side join
  home_name?: string;
}

/** Source: model_versions joined with models */
export interface ModelVersionRow {
  id: string;
  model_id: string;
  version: string;
  status: string;               // active | staging | archived
  created_at: string;
  author: string | null;
  accuracy: number | null;      // 0–1 float
  f1_score: number | null;
  precision: number | null;
  recall: number | null;
  false_positive_rate: number | null;
  latency_ms: number | null;
  memory_mb: number | null;
  throughput_per_s: number | null;
  // client-side join
  model_name?: string;
}

/** Source: edge_nodes joined with homes */
export interface FleetHealthRow {
  id: string;             // edge_nodes.id (uuid PK)
  node_code: string;
  home_id: string;
  status: string;         // online | offline | warning
  ip_address: string | null;
  last_seen_at: string | null;
  current_cpu_percent: number | null;
  current_ram_percent: number | null;
  current_temperature_c: number | null;
  current_latency_ms: number | null;
  framework: string | null;
  model_version_text: string | null;
  // client-side join
  home_name?: string;
}

export interface ReportSummaryStats {
  totalAlerts: number;
  criticalAlerts: number;
  resolvedAlerts: number;
  totalFlows: number;
  anomalyFlows: number;
  retrainJobs: number;
  avgModelAccuracy: number;   // 0–100 percentage
  onlineNodes: number;
  totalNodes: number;
}

export interface DateRange {
  from: string;   // 'YYYY-MM-DD' — empty string = no filter applied
  to: string;
}

export interface ReportFilters {
  dateRange: DateRange;
  homeId: string;
  sections: {
    attackSummary: boolean;
    networkFlows: boolean;
    modelAccuracy: boolean;
    retrainJobs: boolean;
    fleetHealth: boolean;
    auditTrail: boolean;
  };
}

// Backward-compatibility alias
export type AlertSummaryRow = AlertRow;
