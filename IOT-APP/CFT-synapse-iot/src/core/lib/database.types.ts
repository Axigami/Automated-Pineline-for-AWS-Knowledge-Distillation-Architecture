/**
 * database.types.ts – Typescript types khớp với schema Supabase PostgreSQL.
 * Được map thủ công từ Database_App_IoT.sql (9 tables).
 */

export type Database = {
    public: {
        Tables: {
            homes: {
                Row: {
                    id: string;
                    code: string;
                    name: string;
                    region: string | null;
                    created_at: string;
                    cloud_verification_confidence_threshold: number | null;
                    data_drift_alert_level: number | null;
                };
                Insert: Partial<Database['public']['Tables']['homes']['Row']>;
                Update: Partial<Database['public']['Tables']['homes']['Row']>;
            };
            edge_nodes: {
                Row: {
                    id: string;
                    home_id: string;
                    node_code: string;
                    status: string;
                    location_text: string | null;
                    ip_address: string | null;
                    last_seen_at: string | null;
                    deployed_model_version_id: string | null;
                    created_at: string;
                    current_cpu_percent: number | null;
                    current_ram_percent: number | null;
                    current_temperature_c: number | null;
                    current_latency_ms: number | null;
                    framework: string | null;
                    model_version_text: string | null;
                };
                Insert: Partial<Database['public']['Tables']['edge_nodes']['Row']>;
                Update: Partial<Database['public']['Tables']['edge_nodes']['Row']>;
            };
            node_telemetry: {
                Row: {
                    id: string;
                    node_id: string;
                    ts: string;
                    cpu_percent: number | null;
                    ram_percent: number | null;
                    temperature_c: number | null;
                    latency_ms: number | null;
                };
                Insert: Partial<Database['public']['Tables']['node_telemetry']['Row']>;
                Update: Partial<Database['public']['Tables']['node_telemetry']['Row']>;
            };
            alerts_all: {
                Row: {
                    alert_id: string;
                    alert_home_id: string;
                    alert_node_id: string | null;
                    alert_first_seen_at: string;
                    alert_threat_type: string;
                    alert_severity: string;
                    alert_status: string;
                    alert_confidence: number | null;
                    alert_predicted_label: string | null;
                    alert_source_ip: string | null;
                    alert_target_ip: string | null;
                    alert_source_text: string | null;
                    alert_verdict_text: string | null;
                    alert_sequence_values_json: string | null;
                    alert_sequence_steps_json: string | null;
                    alert_created_at: string;
                    audit_action: string | null;
                    audit_target: string | null;
                    audit_created_at: string | null;
                    audit_user_display_name: string | null;
                    audit_status: string | null;
                };
                Insert: Partial<Database['public']['Tables']['alerts_all']['Row']>;
                Update: Partial<Database['public']['Tables']['alerts_all']['Row']>;
            };
            network_flows_feedback_all: {
                Row: {
                    flow_id: string;
                    flow_home_id: string;
                    flow_node_id: string | null;
                    flow_ts: string;
                    flow_protocol: string | null;
                    flow_src_ip: string | null;
                    flow_dst_ip: string | null;
                    flow_src_port: number | null;
                    flow_dst_port: number | null;
                    flow_duration_s: number | null;
                    flow_in_bytes: number | null;
                    flow_out_bytes: number | null;
                    flow_tcp_flags: string | null;
                    flow_total_bytes: number | null;
                    predicted_label: string | null;
                    confidence: number | null;
                    anomaly_score: number | null;
                    is_anomaly: boolean | null;
                    inference_logic: string | null;
                    feedback_action: string | null;
                    feedback_true_label: string | null;
                    feedback_note: string | null;
                    feedback_user_id: string | null;
                    feedback_created_at: string | null;
                };
                Insert: Partial<Database['public']['Tables']['network_flows_feedback_all']['Row']>;
                Update: Partial<Database['public']['Tables']['network_flows_feedback_all']['Row']>;
            };
            model_versions: {
                Row: {
                    id: string;
                    model_id: string;
                    version: string;
                    status: string;
                    artifact_uri: string | null;
                    metrics_json: string | null;
                    created_at: string;
                    author: string | null;
                    accuracy: number | null;
                    f1_score: number | null;
                    precision: number | null;
                    recall: number | null;
                    latency_ms: number | null;
                    memory_mb: number | null;
                    false_positive_rate: number | null;
                    throughput_per_s: number | null;
                };
                Insert: Partial<Database['public']['Tables']['model_versions']['Row']>;
                Update: Partial<Database['public']['Tables']['model_versions']['Row']>;
            };
            retrain_jobs_all: {
                Row: {
                    job_id: string;
                    job_requested_by: string | null;
                    job_home_id: string | null;
                    job_status: string | null;
                    job_data_range: string | null;
                    job_epochs: number | null;
                    job_knowledge_distillation: boolean | null;
                    job_progress_percent: number | null;
                    job_started_at: string | null;
                    job_finished_at: string | null;
                    job_created_at: string | null;
                    job_pipeline_steps_json: string | null;
                    job_training_batch_size: number | null;
                    job_training_learning_rate: number | null;
                    audit_action: string | null;
                    audit_target: string | null;
                    audit_created_at: string | null;
                    audit_user_display_name: string | null;
                    audit_status: string | null;
                };
                Insert: Partial<Database['public']['Tables']['retrain_jobs_all']['Row']>;
                Update: Partial<Database['public']['Tables']['retrain_jobs_all']['Row']>;
            };
            deployments_all: {
                Row: {
                    deployment_id: string;
                    deployment_requested_by: string | null;
                    deployment_model_version_id: string | null;
                    deployment_status: string | null;
                    deployment_created_at: string | null;
                    target_node_id: string | null;
                    target_status: string | null;
                    target_message: string | null;
                    audit_action: string | null;
                    audit_target: string | null;
                    audit_created_at: string | null;
                    audit_user_display_name: string | null;
                    audit_status: string | null;
                };
                Insert: Partial<Database['public']['Tables']['deployments_all']['Row']>;
                Update: Partial<Database['public']['Tables']['deployments_all']['Row']>;
            };
            users_roles_settings: {
                Row: {
                    user_id: string;
                    user_email: string | null;
                    user_display_name: string | null;
                    role_code: string | null;
                    role_name: string | null;
                    setting_home_id: string | null;
                    setting_key: string | null;
                    setting_value_number: number | null;
                    setting_value_text: string | null;
                };
                Insert: Partial<Database['public']['Tables']['users_roles_settings']['Row']>;
                Update: Partial<Database['public']['Tables']['users_roles_settings']['Row']>;
            };
        };
        Views: Record<string, never>;
        Functions: Record<string, never>;
        Enums: Record<string, never>;
    };
};
