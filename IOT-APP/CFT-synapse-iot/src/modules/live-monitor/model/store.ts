/**
 * Live Monitor – Store
 * Quản lý state cho tất cả dữ liệu realtime của module.
 * FR 1.1: Buffer giới hạn BUFFER_LIMIT entries cho mỗi stream.
 */
import { useState, useCallback } from 'react';
import type { LogEntry, FlowUIModel, AlertUIModel, EdgeSyncStatus, RadarPoint } from './types';

const BUFFER_LIMIT = 1000;
const FLOW_DISPLAY_LIMIT = 20;   // hiển thị tối đa 20 rows trong bảng
const RADAR_WINDOW = 30;          // 30 điểm trên chart anomaly score

/**
 * Radar khởi tạo bằng 0 – chỉ nhận dữ liệu thực từ DB.
 * Không dùng mock/random ở bất kỳ đâu.
 */
const EMPTY_RADAR_DATA: import('./types').RadarPoint[] = Array.from(
  { length: RADAR_WINDOW },
  (_, i) => ({ time: i, score: 0 })
);

// ─────────────── Pure helper: apply 1 alert to state (tránh React batch bug với polling) ───────────────

function applyAlertToState(prev: AlertUIModel[], alert: AlertUIModel): AlertUIModel[] {
  // Check for duplicates by ID
  if (prev.some(a => a.id === alert.id)) {
    return prev;
  }

  // Group DoS/DDoS alerts by source IP
  const isDoSOrDDoS = alert.threat && alert.threat.toLowerCase().includes('dos');
  const srcIp = alert.srcIp;
  
  if (isDoSOrDDoS && srcIp) {
    // Find any existing alert (aggregated or not) with same IP within 5 min window
    const existingIdx = prev.findIndex(a => 
      a.srcIp === srcIp && 
      a.threat && a.threat.toLowerCase().includes('dos') &&
      (Date.now() - new Date(a.time).getTime() < 300000)
    );

    if (existingIdx !== -1) {
      const next = [...prev];
      const existing = next[existingIdx];
      
      if (existing.isAggregated) {
        // Merge into existing group
        next[existingIdx] = {
          ...existing,
          time: alert.time,
          timeDisplay: alert.timeDisplay,
          aggregatedCount: (existing.aggregatedCount || 1) + 1,
          aggregatedAlerts: [alert, ...(existing.aggregatedAlerts || [])],
          confidence: alert.confidence,
          sequence: alert.sequence,
          alert_sequence_values_json: alert.alert_sequence_values_json,
        };
      } else {
        // Convert single alert + new alert into a group
        next[existingIdx] = {
          ...existing,
          isAggregated: true,
          aggregatedCount: 2,
          aggregatedAlerts: [existing, alert],
        };
      }
      return next;
    }
  }

  const next = [alert, ...prev];
  return next.length > 50 ? next.slice(0, 50) : next;
}

export function useLiveMonitorStore() {
  // ─── Legacy log entries (backward compat) ───
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [verifyResult, setVerifyResult] = useState<unknown>(null);
  const [isVerifying, setIsVerifying] = useState(false);

  // ─── Flows: Raw data từ Raspberry Pi ───
  const [flows, setFlows] = useState<FlowUIModel[]>([]);

  // ─── Alerts: Cảnh báo từ Cloud model ───
  const [alerts, setAlerts] = useState<AlertUIModel[]>([]);

  // ─── Edge Sync Status ───
  const [syncStatuses, setSyncStatuses] = useState<Record<string, EdgeSyncStatus>>({});

  // ─── Radar: Anomaly score time series (Sliding Window) ───
  // Khởi tạo bằng 0 – sẽ được cập nhật từ DB thực khi mount
  const [radarData, setRadarData] = useState<RadarPoint[]>(EMPTY_RADAR_DATA);

  // ─── Bộ đếm cho thống kê ───
  const [totalFlowsReceived, setTotalFlowsReceived] = useState(0);

  // ─────────────── Actions ───────────────

  /** Thêm 1 flow mới vào đầu danh sách (newest first) */
  const addFlow = useCallback((flow: FlowUIModel) => {
    if (isPaused) return;
    setFlows((prev) => {
      // Always add flow - allow duplicates for real-time visibility
      const next = [flow, ...prev];
      return next.length > BUFFER_LIMIT ? next.slice(0, BUFFER_LIMIT) : next;
    });
    setTotalFlowsReceived((n) => n + 1);
    // Cập nhật radar data chỉ với dữ liệu thực từ DB:
    //   - Ưu tiên anomaly_score (0.0–1.0) nếu DB có ghi
    //   - Nếu không có anomaly_score: dùng confidence khi là anomaly, 0 khi bình thường
    const score =
      flow.anomalyScore !== null
        ? flow.anomalyScore
        : flow.isAnomaly
        ? (flow.confidence ?? 0.5)
        : 0;
    setRadarData((prev) => {
      const last = prev[prev.length - 1];
      return [...prev.slice(1), { time: (last?.time ?? 0) + 1, score }];
    });
  }, [isPaused]);

  /** Bulk load flows khi khởi tạo (lịch sử gần nhất) */
  const setInitialFlows = useCallback((initialFlows: FlowUIModel[]) => {
    setFlows(initialFlows.slice(0, FLOW_DISPLAY_LIMIT));

    // Xây dựng radar từ lịch sử DB thực – không dùng random.
    // Các slot chưa có dữ liệu (padding bên trái) = score 0
    const historical = initialFlows.slice(0, RADAR_WINDOW).reverse();
    const paddedLen = RADAR_WINDOW - historical.length;
    const padPoints: RadarPoint[] = Array.from({ length: paddedLen }, (_, i) => ({
      time: i,
      score: 0, // chưa có dữ liệu
    }));
    const histPoints: RadarPoint[] = historical.map((f, i) => ({
      time: paddedLen + i,
      // Lấy đúng thứ tự ưu tiên: anomaly_score → confidence (nếu anomaly) → 0
      score: f.anomalyScore !== null
        ? f.anomalyScore
        : f.isAnomaly
        ? (f.confidence ?? 0.5)
        : 0,
    }));
    setRadarData([...padPoints, ...histPoints]);
    setTotalFlowsReceived(initialFlows.length);
  }, []);

  /** Thêm 1 alert mới vào đầu danh sách */
  const addAlert = useCallback((alert: AlertUIModel) => {
    // Load alerts from last 24 hours (same as Live Alert Feed)
    const twentyFourHoursAgo = Date.now() - 24 * 60 * 60 * 1000;
    const alertTime = new Date(alert.time).getTime();
    
    if (alertTime < twentyFourHoursAgo) {
      console.log(`[LiveMonitor] Ignoring old alert (${alert.id}) - older than 24 hours`);
      return;
    }
    
    setAlerts((prev) => applyAlertToState(prev, alert));
  }, []);

  /** Thêm nhiều alert cùng lúc (dùng trong polling để tránh React batch bug) */
  const addAlertsBatch = useCallback((newAlerts: AlertUIModel[]) => {
    setAlerts((prev) => {
      let current = prev;
      for (const alert of newAlerts) {
        current = applyAlertToState(current, alert);
      }
      return current;
    });
  }, []);

  /** Load danh sách alerts ban đầu */
  const setInitialAlerts = useCallback((initialAlerts: AlertUIModel[]) => {
    // Load alerts from last 24 hours (same as Live Alert Feed)
    const twentyFourHoursAgo = Date.now() - 24 * 60 * 60 * 1000;
    const recentAlerts = initialAlerts.filter(alert => {
      const alertTime = new Date(alert.time).getTime();
      return alertTime >= twentyFourHoursAgo;
    });
    
    // Group them initially
    const grouped: AlertUIModel[] = [];
    const dosGroups = new Map<string, AlertUIModel>();

    recentAlerts.forEach(alert => {
      const isDoSOrDDoS = alert.threat.toLowerCase().includes('dos');
      if (isDoSOrDDoS && alert.srcIp) {
        const key = alert.srcIp;
        if (dosGroups.has(key)) {
          const group = dosGroups.get(key)!;
          group.aggregatedCount = (group.aggregatedCount || 1) + 1;
          group.aggregatedAlerts = group.aggregatedAlerts || [];
          group.aggregatedAlerts.push(alert);
        } else {
          dosGroups.set(key, { ...alert, isAggregated: true, aggregatedCount: 1, aggregatedAlerts: [alert] });
          grouped.push(dosGroups.get(key)!);
        }
      } else {
        grouped.push(alert);
      }
    });

    console.log(`[LiveMonitor] Filtered and Grouped alerts: ${initialAlerts.length} → ${grouped.length}`);
    setAlerts(grouped);
  }, []);

  /** Cập nhật trạng thái của 1 alert (verifying → confirmed/false-positive) */
  const updateAlertStatus = useCallback((
    id: string,
    status: AlertUIModel['status'],
    verdict?: string
  ) => {
    setAlerts((prev) => {
      // CRITICAL FIX: Remove verified alerts from display after 5 seconds
      if (status === 'confirmed' || status === 'false-positive') {
        setTimeout(() => {
          setAlerts((current) => {
            const filtered = current.filter(a => a.id !== id);
            if (filtered.length < current.length) {
              console.log(`[LiveMonitor] Removed verified alert: ${id}`);
            }
            return filtered;
          });
        }, 5000); // Remove after 5 seconds
      }
      
      return prev.map((a) =>
        a.id === id
          ? { ...a, status, ...(verdict ? { verdict } : {}) }
          : a
      );
    });
  }, []);

  /** Merge alert sau UPDATE từ Supabase (verify từ tab khác / replication) */
  const upsertAlert = useCallback((alert: AlertUIModel) => {
    setAlerts((prev) => {
      const idx = prev.findIndex((a) => a.id === alert.id);
      if (idx === -1) {
        const next = [alert, ...prev];
        return next.length > 50 ? next.slice(0, 50) : next;
      }
      const next = [...prev];
      next[idx] = alert;
      return next;
    });
  }, []);

  /** Legacy addLog */
  const addLog = useCallback((entry: LogEntry) => {
    if (isPaused) return;
    setLogs((prev) => {
      const next = [entry, ...prev];
      return next.length > BUFFER_LIMIT ? next.slice(0, BUFFER_LIMIT) : next;
    });
  }, [isPaused]);

  return {
    // Legacy
    logs, addLog,
    // Flow stream
    flows, addFlow, setInitialFlows,
    // Alert stream
    alerts, setAlerts, addAlert, addAlertsBatch, setInitialAlerts, updateAlertStatus, upsertAlert,
    // Edge sync
    syncStatuses, setSyncStatuses,
    // Radar chart
    radarData, setRadarData,
    // Stats
    totalFlowsReceived,
    // Control
    isPaused, setIsPaused,
    error, setError,
    verifyResult, setVerifyResult,
    isVerifying, setIsVerifying,
  };
}
