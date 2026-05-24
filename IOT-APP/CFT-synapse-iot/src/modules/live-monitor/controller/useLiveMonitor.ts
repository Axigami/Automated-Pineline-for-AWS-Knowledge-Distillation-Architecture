/**
 * useLiveMonitor – Controller hook (Realtime)
 *
 * ┌─────────────────────────────────────────────────────────────┐
 * │  Nguồn dữ liệu realtime:                                    │
 * │  1. network_flows_feedback_all → INSERT                     │
 * │     (Model Edge ở Raspberry Pi gửi lên, mỗi lần có flow)   │
 * │     → Cập nhật: Dual-Stream Raw Data Table + Radar          │
 * │                                                             │
 * │  2. alerts_all → INSERT                                     │
 * │     (Model Cloud CNN-LSTM phát hiện anomaly, ghi alert)     │
 * │     → Cập nhật: Live Alert Stack                            │
 * │                                                             │
 * │  3. edge_nodes → UPDATE                                     │
 * │     (Pi sync lên CPU, RAM, Temp, Latency)                   │
 * │     → Cập nhật: Edge-to-Cloud Sync panel                    │
 * └─────────────────────────────────────────────────────────────┘
 *
 * FR 1.1: Buffer giới hạn 1000 log entries.
 */
import { useEffect, useCallback, useRef, useState } from 'react';
import { supabase } from '../../../core/lib/supabaseClient';
import { useLiveMonitorStore } from '../model/store';
import { adaptFlowRow, adaptAlertRow, adaptEdgeNodeRow } from '../model/adapter';
import type { FlowRow, AlertRow, EdgeNodeRow, AlertUIModel } from '../model/types';

// Số record tải lần đầu (lịch sử gần nhất)
const INITIAL_FLOW_LIMIT = 50;
const INITIAL_ALERT_LIMIT = 20;

/** Chuẩn hóa confidence từ DB (0–1 hoặc 0–100). */
function normalizeAlertConfidence(c: number | null | undefined): number | null {
  if (c == null || Number.isNaN(c)) return null;
  if (c > 1) return Math.min(c / 100, 1);
  return Math.min(Math.max(c, 0), 1);
}

export function useLiveMonitor() {
  const store = useLiveMonitorStore();
  const storeRef = useRef(store);
  storeRef.current = store;

  const [verifyBanner, setVerifyBanner] = useState<string | null>(null);
  const verifyBannerTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Refs giữ channel instances để cleanup
  const flowChannelRef = useRef<ReturnType<typeof supabase.channel> | null>(null);
  const alertChannelRef = useRef<ReturnType<typeof supabase.channel> | null>(null);
  const alertUpdateChannelRef = useRef<ReturnType<typeof supabase.channel> | null>(null);
  const edgeChannelRef = useRef<ReturnType<typeof supabase.channel> | null>(null);

  // Refs cho alert queue (batch Realtime alerts cùng lúc để React 18 batch không phá aggregation)
  const alertQueueRef = useRef<AlertUIModel[]>([]);
  const alertFlushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ─────────────────────────────────────────────────────────────
  // INITIAL LOAD: Lấy lịch sử gần nhất khi mount
  // ─────────────────────────────────────────────────────────────

  const loadInitialFlows = useCallback(async () => {
    const { data, error } = await supabase
      .from('network_flows_feedback_all')
      .select(
        'flow_id, flow_home_id, flow_node_id, flow_ts, flow_protocol, flow_src_ip, flow_dst_ip, ' +
        'flow_src_port, flow_dst_port, flow_duration_s, flow_total_bytes, ' +
        'flow_in_bytes, flow_out_bytes, is_anomaly, predicted_label, ' +
        'confidence, anomaly_score, inference_logic'
      )
      .order('flow_ts', { ascending: false })
      .limit(INITIAL_FLOW_LIMIT);

    if (error) {
      store.setError('Failed to load flow history: ' + error.message);
      return;
    }

    const flows = ((data as unknown as FlowRow[]) ?? []).map(adaptFlowRow);
    store.setInitialFlows(flows);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const loadInitialAlerts = useCallback(async () => {
    // Load alerts from last 24 hours (same as Live Alert Feed)
    const twentyFourHoursAgo = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
    
    const { data, error } = await supabase
      .from('alerts_all')
      .select(
        'alert_id, alert_home_id, alert_node_id, alert_first_seen_at, alert_threat_type, alert_severity, ' +
        'alert_status, alert_confidence, alert_source_ip, alert_target_ip, ' +
        'alert_predicted_label, alert_verdict_text, alert_sequence_values_json, ' +
        'alert_class_id'
      )
      .gt('alert_first_seen_at', twentyFourHoursAgo)  // ✅ LAST 24 HOURS (same as Live Alert Feed)
      .not('alert_predicted_label', 'is', null)  // ✅ ONLY ALERTS WITH PREDICTION (model processed)
      .not('alert_confidence', 'is', null)  // ✅ ONLY ALERTS WITH CONFIDENCE SCORE
      .order('alert_first_seen_at', { ascending: false })
      .limit(INITIAL_ALERT_LIMIT);

    if (error) {
      store.setError('Failed to load alerts: ' + error.message);
      return;
    }

    const alerts = ((data as unknown as AlertRow[]) ?? []).map(adaptAlertRow);
    console.log(`[LiveMonitor] Loaded ${alerts.length} recent alerts (last 24 hours, with predictions)`);
    store.setInitialAlerts(alerts);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const loadEdgeNodeStatus = useCallback(async () => {
    // Lấy tất cả các nodes để hiển thị tùy chọn và trạng thái
    const { data, error } = await supabase
      .from('edge_nodes')
      .select(
        'id, node_code, home_id, status, current_cpu_percent, current_ram_percent, ' +
        'current_temperature_c, current_latency_ms, last_seen_at'
      )
      .order('node_code', { ascending: true });

    if (error || !data) return;

    const statusesMap: Record<string, ReturnType<typeof adaptEdgeNodeRow>> = {};
    (data as unknown as EdgeNodeRow[]).forEach(row => {
      statusesMap[row.id] = adaptEdgeNodeRow(row);
    });
    store.setSyncStatuses(statusesMap);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ─────────────────────────────────────────────────────────────
  // REALTIME SUBSCRIPTIONS
  // ─────────────────────────────────────────────────────────────

  useEffect(() => {
    // Bước 1: Load lịch sử ban đầu
    loadInitialFlows();
    loadInitialAlerts();
    loadEdgeNodeStatus();

    console.log('[LiveMonitor] Setting up Realtime subscriptions...');

    // ── Channel 1: network_flows_feedback_all → INSERT ──
    // Dữ liệu từ Raspberry Pi (model Edge) đẩy lên.
    // addFlow() trong store sẽ tự cập nhật radarData với anomaly_score thực từ DB.
    const flowChannel = supabase
      .channel('live_monitor_flows')
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'network_flows_feedback_all' },
        (payload) => {
          console.log('[LiveMonitor] New flow received:', payload);
          if (storeRef.current.isPaused) {
            console.log('[LiveMonitor] Flow ignored - stream is paused');
            return;
          }
          const row = payload.new as FlowRow;
          const flow = adaptFlowRow(row);
          console.log('[LiveMonitor] Adding flow to UI:', flow.id);
          // addFlow cập nhật cả flows list lẫn radarData sliding window
          storeRef.current.addFlow(flow);
        }
      )
      .subscribe((status) => {
        console.log('[LiveMonitor] Flow channel status:', status);
      });

    flowChannelRef.current = flowChannel;

    // ── Channel 2: alerts_all → INSERT ──
    // Dùng queue ref + flush timer để batch alerts đến gần nhau,
    // tránh React 18 batching làm mất aggregation DoS/DDoS
    const flushAlertQueue = () => {
      if (alertQueueRef.current.length > 0) {
        const batch = alertQueueRef.current.splice(0);
        console.log(`[LiveMonitor] Flushing ${batch.length} queued alerts`);
        storeRef.current.addAlertsBatch(batch);
      }
      alertFlushTimerRef.current = null;
    };

    const alertChannel = supabase
      .channel('live_monitor_alerts')
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'alerts_all' },
        (payload) => {
          const row = payload.new as AlertRow;
          const alert = adaptAlertRow(row);
          
          // Queue alert, flush after debounce to batch DoS/DDoS from same IP
          alertQueueRef.current.push(alert);
          if (alertFlushTimerRef.current) clearTimeout(alertFlushTimerRef.current);
          alertFlushTimerRef.current = setTimeout(flushAlertQueue, 200);
        }
      )
      .subscribe((status) => {
        console.log('[LiveMonitor] Alert channel status:', status);
      });

    alertChannelRef.current = alertChannel;

    const alertUpdateChannel = supabase
      .channel('live_monitor_alerts_updates')
      .on(
        'postgres_changes',
        { event: 'UPDATE', schema: 'public', table: 'alerts_all' },
        (payload) => {
          console.log('[LiveMonitor] Alert updated:', payload);
          const row = payload.new as AlertRow;
          const alert = adaptAlertRow(row);
          console.log('[LiveMonitor] Updating alert in UI:', alert.id);
          storeRef.current.upsertAlert(alert);
        }
      )
      .subscribe((status) => {
        console.log('[LiveMonitor] Alert update channel status:', status);
      });

    alertUpdateChannelRef.current = alertUpdateChannel;

    // ── Channel 3: edge_nodes → UPDATE ──
    // Cập nhật CPU/RAM/Temp/Latency từ Pi
    const edgeChannel = supabase
      .channel('live_monitor_edge_nodes')
      .on(
        'postgres_changes',
        { event: 'UPDATE', schema: 'public', table: 'edge_nodes' },
        (payload) => {
          console.log('[LiveMonitor] Edge node updated:', payload);
          const row = payload.new as EdgeNodeRow;
          const syncStatus = adaptEdgeNodeRow(row);
          console.log('[LiveMonitor] Updating edge node status:', syncStatus.nodeCode);
          storeRef.current.setSyncStatuses(prev => ({ ...prev, [row.id]: syncStatus }));
        }
      )
      .subscribe((status) => {
        console.log('[LiveMonitor] Edge node channel status:', status);
      });

    edgeChannelRef.current = edgeChannel;

    console.log('[LiveMonitor] All Realtime subscriptions initialized');

    // ── AUTO-CLEANUP: Remove alerts older than 24 hours every 60 seconds ──
    const cleanupInterval = setInterval(() => {
      const twentyFourHoursAgo = Date.now() - 24 * 60 * 60 * 1000;
      storeRef.current.setAlerts((prev) => {
        const filtered = prev.filter(alert => {
          const alertTime = new Date(alert.time).getTime();
          return alertTime >= twentyFourHoursAgo;
        });
        if (filtered.length < prev.length) {
          console.log(`[LiveMonitor] Auto-cleanup: Removed ${prev.length - filtered.length} old alerts`);
        }
        return filtered;
      });
    }, 60000); // Every 60 seconds

    // ── FALLBACK POLLING: Refresh data every 5 seconds if not paused ──
    // This ensures data is always fresh even if Realtime fails
    const pollingInterval = setInterval(async () => {
      if (storeRef.current.isPaused) return;
      
      console.log('[LiveMonitor] Polling for new data...');
      
      try {
        // Get the most recent flow timestamp we have
        const latestFlow = storeRef.current.flows[0];
        let since: string;
        
        if (latestFlow?.timestamp) {
          try {
            const date = new Date(latestFlow.timestamp);
            if (isNaN(date.getTime())) {
              // Invalid date, use fallback
              since = new Date(Date.now() - 10000).toISOString();
            } else {
              since = date.toISOString();
            }
          } catch {
            // Error parsing date, use fallback
            since = new Date(Date.now() - 10000).toISOString();
          }
        } else {
          // No flows yet, get last 10 seconds
          since = new Date(Date.now() - 10000).toISOString();
        }
        
        // Fetch new flows
        const { data: newFlows, error: flowError } = await supabase
          .from('network_flows_feedback_all')
          .select(
            'flow_id, flow_home_id, flow_node_id, flow_ts, flow_protocol, flow_src_ip, flow_dst_ip, ' +
            'flow_src_port, flow_dst_port, flow_duration_s, flow_total_bytes, ' +
            'flow_in_bytes, flow_out_bytes, is_anomaly, predicted_label, ' +
            'confidence, anomaly_score, inference_logic'
          )
          .gt('flow_ts', since)
          .order('flow_ts', { ascending: false })
          .limit(20);
        
        if (flowError) {
          console.error('[LiveMonitor] Polling flow error:', flowError);
        } else if (newFlows && newFlows.length > 0) {
          console.log(`[LiveMonitor] Polling found ${newFlows.length} new flows`);
          // Reverse to add oldest first (maintains chronological order)
          const flowsToAdd = (newFlows as unknown as FlowRow[]).reverse();
          flowsToAdd.forEach(row => {
            const flow = adaptFlowRow(row);
            storeRef.current.addFlow(flow); // addFlow has duplicate check
          });
        }
        
        // Fetch new alerts
        const latestAlert = storeRef.current.alerts[0];
        let alertSince: string;
        
        // Load alerts from last 24 hours (same as Live Alert Feed)
        const twentyFourHoursAgo = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
        
        if (latestAlert?.time) {
          try {
            const date = new Date(latestAlert.time);
            if (isNaN(date.getTime())) {
              alertSince = twentyFourHoursAgo;
            } else {
              // Use the LATER of: latest alert time OR 24 hours ago
              alertSince = date.toISOString() > twentyFourHoursAgo ? date.toISOString() : twentyFourHoursAgo;
            }
          } catch {
            alertSince = twentyFourHoursAgo;
          }
        } else {
          alertSince = twentyFourHoursAgo;
        }
        
        const { data: newAlerts, error: alertError } = await supabase
          .from('alerts_all')
          .select(
            'alert_id, alert_home_id, alert_node_id, alert_first_seen_at, alert_threat_type, alert_severity, ' +
            'alert_status, alert_confidence, alert_source_ip, alert_target_ip, ' +
            'alert_predicted_label, alert_verdict_text, alert_sequence_values_json, ' +
            'alert_class_id'
          )
          .gt('alert_first_seen_at', alertSince)
          .not('alert_predicted_label', 'is', null)  // ✅ ONLY ALERTS WITH PREDICTION
          .not('alert_confidence', 'is', null)  // ✅ ONLY ALERTS WITH CONFIDENCE
          .order('alert_first_seen_at', { ascending: false })
          .limit(10);
        
        if (alertError) {
          console.error('[LiveMonitor] Polling alert error:', alertError);
        } else if (newAlerts && newAlerts.length > 0) {
          console.log(`[LiveMonitor] Polling found ${newAlerts.length} new alerts`);
          // Use batch to avoid React state update batching issues with DoS/DDoS grouping
          const alertsToAdd = (newAlerts as unknown as AlertRow[])
            .map(adaptAlertRow)
            .filter(alert => {
              const twentyFourHoursAgo = Date.now() - 24 * 60 * 60 * 1000;
              return new Date(alert.time).getTime() >= twentyFourHoursAgo;
            })
            .reverse();
          if (alertsToAdd.length > 0) {
            storeRef.current.addAlertsBatch(alertsToAdd);
          }
        }
      } catch (error) {
        console.error('[LiveMonitor] Polling error:', error);
      }
    }, 2000); // Poll every 2 seconds for faster updates

    // Cleanup: hủy tất cả channels khi unmount
    return () => {
      console.log('[LiveMonitor] Cleaning up Realtime subscriptions...');
      clearInterval(pollingInterval);
      clearInterval(cleanupInterval);
      if (flowChannelRef.current) supabase.removeChannel(flowChannelRef.current);
      if (alertChannelRef.current) supabase.removeChannel(alertChannelRef.current);
      if (alertUpdateChannelRef.current) supabase.removeChannel(alertUpdateChannelRef.current);
      if (edgeChannelRef.current) supabase.removeChannel(edgeChannelRef.current);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ─────────────────────────────────────────────────────────────
  // ACTIONS
  // ─────────────────────────────────────────────────────────────

  /**
   * (FR 1.2) Alert verification flow:
   * 1) Read alert + threshold `homes.cloud_verification_confidence_threshold` to create contextual verdict.
   * 2) POST `/verify-alert` via API Gateway to invoke Lambda (VerifyAlertProxy) for CNN-LSTM re-inference.
   * 3) PATCH alerts_all: alert_status, alert_verified_at, alert_verdict_text (source of truth for dashboard/pipeline).
   */
  const verifyAlert = useCallback(async (alertId: string) => {
    const st = storeRef.current;
    if (verifyBannerTimerRef.current) {
      clearTimeout(verifyBannerTimerRef.current);
      verifyBannerTimerRef.current = null;
    }
    setVerifyBanner(null);

    st.updateAlertStatus(alertId, 'verifying');
    st.setIsVerifying(true);
    st.setError(null);

    const { data: alertRow, error: alertErr } = await supabase
      .from('alerts_all')
      .select('alert_home_id, alert_confidence, alert_threat_type')
      .eq('alert_id', alertId)
      .maybeSingle();

    if (alertErr || !alertRow) {
      const msg = alertErr?.message ?? 'Alert not found';
      st.setError('Cannot verify: ' + msg);
      st.updateAlertStatus(alertId, 'pending');
      st.setIsVerifying(false);
      return;
    }

    const homeId = (alertRow as { alert_home_id?: string }).alert_home_id;

    // Call Lambda via API Gateway
    const {
      data: { session },
    } = await supabase.auth.getSession();
    
    let lambdaSuccess = false;
    let lambdaMessage = '';
    
    if (session?.access_token) {
      try {
        const { lambdaClient } = await import('../../../core/lib/lambdaClient');
        lambdaClient.setAuthToken(session.access_token);
        lambdaSuccess = await lambdaClient.verifyCloudAlert({ alert_id: alertId, home_id: homeId });
        
        if (lambdaSuccess) {
          lambdaMessage = 'Alert verified successfully via CNN-LSTM teacher model.';
        } else {
          lambdaMessage = 'Lambda verify-alert endpoint returned error or not found.';
        }
      } catch (error) {
        lambdaMessage = `Lambda call failed: ${error instanceof Error ? error.message : 'Unknown error'}`;
        console.error('[LiveMonitor] Lambda verification error:', error);
      }
    } else {
      lambdaMessage = 'No authentication token available.';
    }

    // Update local UI state
    st.updateAlertStatus(alertId, lambdaSuccess ? 'confirmed' : 'pending');
    st.setIsVerifying(false);

    // Show banner notification
    setVerifyBanner(lambdaMessage);
    verifyBannerTimerRef.current = setTimeout(() => {
      setVerifyBanner(null);
      verifyBannerTimerRef.current = null;
    }, 8000);
  }, []);

  // ─────────────────────────────────────────────────────────────
  // PUBLIC API
  // ─────────────────────────────────────────────────────────────

  return {
    // Flows stream (từ Raspberry Pi)
    flows: store.flows,
    totalFlowsReceived: store.totalFlowsReceived,

    // Alerts stream (từ Cloud model)
    alerts: store.alerts as AlertUIModel[],

    // Edge sync status map
    syncStatuses: store.syncStatuses,

    // Anomaly radar
    radarData: store.radarData,

    // Control
    isPaused: store.isPaused,
    togglePause: () => store.setIsPaused((p: boolean) => !p),

    // Actions
    verifyAlert,

    // Status
    error: store.error,
    isVerifying: store.isVerifying,
    verifyBanner,
    dismissVerifyBanner: () => {
      if (verifyBannerTimerRef.current) clearTimeout(verifyBannerTimerRef.current);
      verifyBannerTimerRef.current = null;
      setVerifyBanner(null);
    },

    // Legacy (backward compat)
    logs: store.logs,
  };
}
