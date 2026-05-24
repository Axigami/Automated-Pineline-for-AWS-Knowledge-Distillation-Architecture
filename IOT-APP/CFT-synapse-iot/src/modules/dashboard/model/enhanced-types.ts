/**
 * Enhanced type definitions for Dashboard UI Improvements
 * 
 * This file contains new types for:
 * - Alert aggregation
 * - Fleet node health metrics
 * - Home/device hierarchy
 * - Notification sound control
 */

import { AlertUIModel, EdgeNodeRow, HomeInfo } from './types';

// ============================================================================
// Alert Aggregation Types
// ============================================================================

/**
 * Aggregated alert group containing multiple similar alerts
 */
export interface AggregatedAlert {
  /** Group ID (generated) or single alert ID */
  id: string;
  
  /** Whether this is an aggregated group or single alert */
  isGroup: boolean;
  
  /** Number of alerts in this group */
  count: number;
  
  /** All alerts in this group */
  alerts: AlertUIModel[];
  
  /** Source IP address (common to all alerts in group) */
  sourceIp: string;
  
  /** Target IP address (from first alert) */
  targetIp: string;
  
  /** Attack type (common to all alerts in group) */
  attackType: string;
  
  /** Severity level */
  severity: 'high' | 'medium' | 'low';
  
  /** Timestamp of first alert in group */
  firstSeen: string;
  
  /** Timestamp of most recent alert in group */
  lastSeen: string;
  
  /** Alert status */
  status: string;
}

/**
 * Configuration for alert aggregation algorithm
 */
export interface AggregationConfig {
  /** Time window in milliseconds (default: 5 minutes = 300000ms) */
  timeWindowMs: number;
  
  /** Fields to group by (default: ['sourceIp', 'attackType']) */
  groupByFields: ('sourceIp' | 'attackType')[];
}

// ============================================================================
// Fleet Node Health Types
// ============================================================================

/**
 * Node operational status
 */
export type NodeStatus = 'online' | 'offline' | 'error';

/**
 * Node monitoring status
 */
export type MonitoringStatus = 'active' | 'inactive' | 'paused';

/**
 * Health metric status based on thresholds
 */
export type HealthStatus = 'healthy' | 'warning' | 'critical';

/**
 * Individual health metric with value and status
 */
export interface HealthMetric {
  /** Current value */
  value: number;
  
  /** Unit of measurement */
  unit: string;
  
  /** Health status based on thresholds */
  status: HealthStatus;
  
  /** Thresholds for warning and critical states */
  threshold: {
    warning: number;
    critical: number;
  };
}

/**
 * Complete health metrics for a fleet node
 */
export interface NodeHealthMetrics {
  /** Node ID */
  nodeId: string;
  
  /** Node code identifier */
  nodeCode: string;
  
  /** Operational status */
  status: NodeStatus;
  
  /** Monitoring status */
  monitoringStatus: MonitoringStatus;
  
  /** Health metrics */
  health: {
    cpu: HealthMetric;
    ram: HealthMetric;
    temperature: HealthMetric;
    latency: HealthMetric;
  };
  
  /** Last seen timestamp */
  lastSeen: Date;
  
  /** Node location */
  location: string;
  
  /** Node IP address */
  ipAddress: string;
  
  /** Error message (if status is 'error') */
  errorMessage?: string;
}

// ============================================================================
// Home and Device Types
// ============================================================================

/**
 * Device information
 */
export interface DeviceInfo {
  /** Device ID */
  id: string;
  
  /** Associated home ID */
  home_id: string;
  
  /** Device IP address */
  ip_address: string;
  
  /** Device type (e.g., 'camera', 'sensor', 'router') */
  device_type: string;
  
  /** Device status */
  status: 'active' | 'inactive';
  
  /** Last seen timestamp */
  last_seen_at: string | null;
}

/**
 * Home with associated devices and fleet nodes
 */
export interface HomeWithDevices {
  /** Home information */
  home: HomeInfo;
  
  /** Devices in this home */
  devices: DeviceInfo[];
  
  /** Fleet nodes assigned to this home */
  fleetNodes: EdgeNodeRow[];
  
  /** Count of recent alerts for this home */
  alertCount: number;
  
  /** Warning flag (true if no fleet nodes assigned) */
  hasWarning: boolean;
}

// ============================================================================
// Notification Sound Types
// ============================================================================

/**
 * Configuration for notification sound controller
 */
export interface SoundConfig {
  /** Debounce time in milliseconds (default: 3000ms) */
  debounceMs: number;
  
  /** Path to sound file */
  soundFile: string;
  
  /** Volume level (0.0 to 1.0) */
  volume: number;
}

/**
 * Notification sound controller interface
 */
export interface NotificationSoundController {
  /** Play notification sound (with debouncing) */
  playSound(): void;
  
  /** Enable or disable notification sounds */
  setEnabled(enabled: boolean): void;
  
  /** Check if notification sounds are enabled */
  isEnabled(): boolean;
}

// ============================================================================
// Real-time Update Types
// ============================================================================

/**
 * Real-time connection status
 */
export type ConnectionStatus = 'connected' | 'disconnected' | 'reconnecting';

/**
 * Configuration for real-time update manager
 */
export interface RealtimeConfig {
  /** Tables to subscribe to */
  tables: string[];
  
  /** Fallback polling interval in milliseconds (default: 5000ms) */
  fallbackPollIntervalMs: number;
  
  /** Reconnection delay in milliseconds (default: 2000ms) */
  reconnectDelayMs: number;
  
  /** Maximum reconnection delay in milliseconds (default: 30000ms) */
  maxReconnectDelayMs: number;
}

/**
 * Real-time update manager interface
 */
export interface RealtimeUpdateManager {
  /** Subscribe to table changes */
  subscribe(tables: string[], callback: () => void): () => void;
  
  /** Get current connection status */
  getConnectionStatus(): ConnectionStatus;
  
  /** Get last update timestamp */
  getLastUpdateTime(): Date | null;
}

// ============================================================================
// Error Handling Types
// ============================================================================

/**
 * Error severity level
 */
export type ErrorSeverity = 'low' | 'medium' | 'high';

/**
 * Error context for error handling
 */
export interface ErrorContext {
  /** Operation that failed */
  operation: string;
  
  /** Error severity */
  severity: ErrorSeverity;
  
  /** Whether the operation can be retried */
  retryable: boolean;
  
  /** User-friendly error message */
  userMessage: string;
}

/**
 * Error handler interface
 */
export interface ErrorHandler {
  /** Handle an error with context */
  handleError(error: Error, context: ErrorContext): void;
}

// ============================================================================
// Enhanced Alert UI Model
// ============================================================================

/**
 * Extended alert UI model with aggregation fields
 */
export interface EnhancedAlertUIModel extends AlertUIModel {
  /** Whether this alert is part of an aggregated group */
  isAggregated: boolean;
  
  /** Count of alerts in aggregated group (1 if not aggregated) */
  aggregatedCount: number;
  
  /** First seen timestamp (for aggregated groups) */
  firstSeenAt: string;
  
  /** Last seen timestamp (for aggregated groups) */
  lastSeenAt: string;
  
  /** Child alerts if this is an aggregated group */
  aggregatedAlerts?: AlertUIModel[];
  
  /** Target IP address */
  targetIp: string;
}
