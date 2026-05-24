CREATE TABLE "homes" (
  "id" uuid PRIMARY KEY,
  "code" varchar UNIQUE NOT NULL,
  "name" varchar NOT NULL,
  "region" varchar,
  "created_at" timestamp NOT NULL,
  "cloud_verification_confidence_threshold" int,
  "data_drift_alert_level" float
);

CREATE TABLE "edge_nodes" (
  "id" uuid PRIMARY KEY,
  "home_id" uuid NOT NULL,
  "node_code" varchar UNIQUE NOT NULL,
  "status" varchar NOT NULL,
  "location_text" varchar,
  "ip_address" varchar,
  "last_seen_at" timestamp,
  "deployed_model_version_id" uuid,
  "created_at" timestamp NOT NULL,
  "current_cpu_percent" float,
  "current_ram_percent" float,
  "current_temperature_c" float,
  "current_latency_ms" int,
  "framework" varchar,
  "model_version_text" varchar
);

CREATE TABLE "node_telemetry" (
  "id" uuid PRIMARY KEY,
  "node_id" uuid NOT NULL,
  "ts" timestamp NOT NULL,
  "cpu_percent" float,
  "ram_percent" float,
  "temperature_c" float,
  "latency_ms" int
);

CREATE TABLE "models" (
  "id" uuid PRIMARY KEY,
  "name" varchar NOT NULL,
  "kind" varchar NOT NULL,
  "description" varchar,
  "created_at" timestamp NOT NULL
);

CREATE TABLE "model_versions" (
  "id" uuid PRIMARY KEY,
  "model_id" uuid NOT NULL,
  "version" varchar NOT NULL,
  "status" varchar NOT NULL,
  "artifact_uri" varchar,
  "metrics_json" text,
  "created_at" timestamp NOT NULL,
  "author" varchar,
  "accuracy" float,
  "f1_score" float,
  "precision" float,
  "recall" float,
  "latency_ms" int,
  "memory_mb" int,
  "false_positive_rate" float,
  "throughput_per_s" float
);

CREATE TABLE "flow_inference" (
  "id" uuid PRIMARY KEY,
  "flow_id" uuid NOT NULL,
  "model_version_id" uuid NOT NULL,
  "engine" varchar NOT NULL,
  "predicted_label" varchar,
  "confidence" float,
  "anomaly_score" float,
  "is_anomaly" boolean,
  "created_at" timestamp NOT NULL
);

CREATE TABLE "users_roles_settings" (
  "user_id" uuid PRIMARY KEY,
  "user_email" varchar,
  "user_display_name" varchar,
  "user_password_hash" varchar,
  "user_created_at" timestamp,
  "role_id" uuid,
  "role_code" varchar,
  "role_name" varchar,
  "user_role_user_id" uuid,
  "user_role_role_id" uuid,
  "setting_id" uuid,
  "setting_home_id" uuid,
  "setting_key" varchar,
  "setting_value_number" float,
  "setting_value_text" varchar,
  "setting_updated_by" uuid,
  "setting_updated_at" timestamp
);

CREATE TABLE "deployments_all" (
  "deployment_id" uuid PRIMARY KEY,
  "deployment_requested_by" uuid,
  "deployment_model_version_id" uuid,
  "deployment_status" varchar,
  "deployment_created_at" timestamp,
  "target_id" uuid,
  "target_deployment_id" uuid,
  "target_node_id" uuid,
  "target_status" varchar,
  "target_message" varchar,
  "audit_user_id" uuid,
  "audit_user_email" varchar,
  "audit_user_display_name" varchar,
  "audit_action" varchar,
  "audit_target" varchar,
  "audit_status" varchar,
  "audit_created_at" timestamp
);

CREATE TABLE "network_flows_feedback_all" (
  "flow_id" uuid PRIMARY KEY,
  "flow_home_id" uuid NOT NULL,
  "flow_node_id" uuid,
  "flow_ts" timestamp NOT NULL,
  "flow_protocol" varchar,
  "flow_src_ip" varchar,
  "flow_dst_ip" varchar,
  "flow_src_port" int,
  "flow_dst_port" int,
  "flow_duration_s" float,
  "flow_in_bytes" bigint,
  "flow_out_bytes" bigint,
  "flow_tcp_flags" varchar,
  "flow_created_at" timestamp NOT NULL,
  "flow_total_bytes" bigint,
  "flow_length" int,
  "feedback_id" uuid,
  "feedback_flow_id" uuid,
  "feedback_user_id" uuid,
  "feedback_action" varchar,
  "feedback_true_label" varchar,
  "feedback_note" varchar,
  "feedback_created_at" timestamp,
  "predicted_label" varchar,
  "confidence" float,
  "anomaly_score" float,
  "is_anomaly" boolean,
  "inference_logic" varchar
);

CREATE TABLE "retrain_jobs_all" (
  "job_id" uuid PRIMARY KEY,
  "job_requested_by" uuid,
  "job_home_id" uuid,
  "job_status" varchar,
  "job_data_range" varchar,
  "job_epochs" int,
  "job_knowledge_distillation" boolean,
  "job_teacher_from_version_id" uuid,
  "job_teacher_to_version_id" uuid,
  "job_student_from_version_id" uuid,
  "job_student_to_version_id" uuid,
  "job_progress_percent" int,
  "job_started_at" timestamp,
  "job_finished_at" timestamp,
  "job_created_at" timestamp,
  "event_id" uuid,
  "event_job_id" uuid,
  "event_ts" timestamp,
  "event_step" varchar,
  "event_message" varchar,
  "event_progress_percent" int,
  "job_pipeline_steps_json" text,
  "job_training_batch_size" int,
  "job_training_learning_rate" float,
  "audit_user_id" uuid,
  "audit_user_email" varchar,
  "audit_user_display_name" varchar,
  "audit_action" varchar,
  "audit_target" varchar,
  "audit_status" varchar,
  "audit_created_at" timestamp
);

CREATE TABLE "alerts_all" (
  "alert_id" uuid PRIMARY KEY,
  "alert_home_id" uuid NOT NULL,
  "alert_node_id" uuid,
  "alert_first_seen_at" timestamp NOT NULL,
  "alert_threat_type" varchar NOT NULL,
  "alert_severity" varchar NOT NULL,
  "alert_status" varchar NOT NULL,
  "alert_edge_model_version_id" uuid,
  "alert_cloud_model_version_id" uuid,
  "alert_verified_at" timestamp,
  "alert_verdict_text" varchar,
  "alert_created_at" timestamp NOT NULL,
  "seq_id" uuid,
  "seq_alert_id" uuid,
  "seq_index" int,
  "seq_value" float,
  "alert_confidence" float,
  "alert_predicted_label" varchar,
  "alert_source_ip" varchar,
  "alert_target_ip" varchar,
  "alert_source_text" varchar,
  "alert_class_id" int,
  "alert_sequence_values_json" text,
  "alert_sequence_steps_json" text,
  "audit_user_id" uuid,
  "audit_user_email" varchar,
  "audit_user_display_name" varchar,
  "audit_action" varchar,
  "audit_target" varchar,
  "audit_status" varchar,
  "audit_created_at" timestamp
);

CREATE UNIQUE INDEX ON "model_versions" ("model_id", "version");

ALTER TABLE "edge_nodes" ADD FOREIGN KEY ("home_id") REFERENCES "homes" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "edge_nodes" ADD FOREIGN KEY ("deployed_model_version_id") REFERENCES "model_versions" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "node_telemetry" ADD FOREIGN KEY ("node_id") REFERENCES "edge_nodes" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "model_versions" ADD FOREIGN KEY ("model_id") REFERENCES "models" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "flow_inference" ADD FOREIGN KEY ("model_version_id") REFERENCES "model_versions" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "users_roles_settings" ADD FOREIGN KEY ("setting_home_id") REFERENCES "homes" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "deployments_all" ADD FOREIGN KEY ("deployment_model_version_id") REFERENCES "model_versions" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "deployments_all" ADD FOREIGN KEY ("target_node_id") REFERENCES "edge_nodes" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "network_flows_feedback_all" ADD FOREIGN KEY ("flow_home_id") REFERENCES "homes" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "network_flows_feedback_all" ADD FOREIGN KEY ("flow_node_id") REFERENCES "edge_nodes" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "retrain_jobs_all" ADD FOREIGN KEY ("job_home_id") REFERENCES "homes" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "retrain_jobs_all" ADD FOREIGN KEY ("job_teacher_from_version_id") REFERENCES "model_versions" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "retrain_jobs_all" ADD FOREIGN KEY ("job_teacher_to_version_id") REFERENCES "model_versions" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "retrain_jobs_all" ADD FOREIGN KEY ("job_student_from_version_id") REFERENCES "model_versions" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "retrain_jobs_all" ADD FOREIGN KEY ("job_student_to_version_id") REFERENCES "model_versions" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "alerts_all" ADD FOREIGN KEY ("alert_home_id") REFERENCES "homes" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "alerts_all" ADD FOREIGN KEY ("alert_node_id") REFERENCES "edge_nodes" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "alerts_all" ADD FOREIGN KEY ("alert_edge_model_version_id") REFERENCES "model_versions" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "alerts_all" ADD FOREIGN KEY ("alert_cloud_model_version_id") REFERENCES "model_versions" ("id") DEFERRABLE INITIALLY IMMEDIATE;
