import { useEffect, useCallback, useRef } from 'react';
import { supabase } from '../../../core/lib/supabaseClient';
import { useAuthContext } from '../../../core/auth/AuthProvider';
import { useDashboardStore } from '../model/store';
import { adaptAlerts, getAttackColor } from '../model/adapter';
import { aggregateAlerts } from '../model/alert-aggregation';
import { useNotificationSound } from '../../../core/hooks/useNotificationSound';
import {
  getDashboardCacheForUser,
  isCacheValid,
  writeCache,
  invalidateCache,
} from '../model/cache';
import type {
  AlertRow, TelemetrySummary, TrafficSeriesPoint,
  EdgeNodeRow, AttackDistPoint, ModelStats, HomeInfo, AlertUIModel,
} from '../model/types';
import type { DashboardCacheState } from '../model/cache';

/**
 * useDashboard – Controller hook với cache theo user + Supabase Realtime.
 *
 * Chiến lược:
 *  1. Mount → nếu cache hợp lệ cho user hiện tại, load từ cache.
 *  2. Ngược lại → fetch, ghi vào cache của user đó.
 *  3. Realtime → invalidate cache user hiện tại & refetch.
 *  4. Đổi user → cache khác user, không dùng chung.
 */
export function useDashboard() {
  const { user } = useAuthContext();
  const userId = user?.id ?? null;
  const userIdRef = useRef(userId);
  userIdRef.current = userId;

  const store = useDashboardStore();
  const {
    setSummary,
    setRecentAlerts,
    setTrafficSeries,
    setHomes,
    setEdgeNodes,
    setAttackDist,
    setModelStats,
    setNetworkFlowMetrics,
    setIsLoading,
    setError,
  } = store;
  const channelRef = useRef<ReturnType<typeof supabase.channel> | null>(null);
  
  // Notification sound with 3-second debouncing
  const { playSound } = useNotificationSound();
  const previousAlertCountRef = useRef<number>(0);
  const lastAlertSourceIpRef = useRef<string | null>(null); // Track last alert source IP

  const syncFromCache = useCallback(
    (uid: string) => {
      const c = getDashboardCacheForUser(uid);
      setSummary(c.summary);
      setRecentAlerts(c.recentAlerts);
      setTrafficSeries(c.trafficSeries);
      setHomes(c.homes);
      setEdgeNodes(c.edgeNodes);
      setAttackDist(c.attackDist);
      setModelStats(c.modelStats);
      setNetworkFlowMetrics(c.networkFlowMetrics || []);
    },
    [setSummary, setRecentAlerts, setTrafficSeries, setHomes, setEdgeNodes, setAttackDist, setModelStats, setNetworkFlowMetrics],
  );

  const fetchSummary = async (nodeIds: string[]): Promise<Partial<{ summary: TelemetrySummary }>> => {
    const todayStart = new Date();
    todayStart.setHours(0, 0, 0, 0);

    // Get node status info
    const { data: nodesData, count: nodesCount } = await supabase
      .from('edge_nodes' as any)
      .select('id, status', { count: 'exact' });

    const nodes = (nodesData ?? []) as any[];
    const totalDevices = nodesCount ?? nodes.length;
    const onlineDevices = nodes.filter((n) => n.status === 'online').length;

    // If no nodes exist, return empty summary
    if (nodeIds.length === 0) {
      return {
        summary: {
          totalDevices: 0,
          onlineDevices: 0,
          alertsToday: 0,
          criticalAlerts: 0,
        },
      };
    }

    // CRITICAL FIX: Only count alerts from nodes in the fleet AND with valid predictions
    const [alertsTodayRes, criticalRes] = await Promise.all([
      supabase
        .from('alerts_all' as any)
        .select('alert_id', { count: 'exact', head: true })
        .gte('alert_first_seen_at', todayStart.toISOString())
        .in('alert_node_id', nodeIds)  // ✅ ONLY ALERTS FROM FLEET NODES
        .not('alert_predicted_label', 'is', null)  // ✅ ONLY ALERTS WITH PREDICTION
        .not('alert_confidence', 'is', null),  // ✅ ONLY ALERTS WITH CONFIDENCE
      supabase
        .from('alerts_all' as any)
        .select('alert_id', { count: 'exact', head: true })
        .gte('alert_first_seen_at', todayStart.toISOString())
        .in('alert_severity', ['critical', 'high'])
        .in('alert_node_id', nodeIds)  // ✅ ONLY ALERTS FROM FLEET NODES
        .not('alert_predicted_label', 'is', null)  // ✅ ONLY ALERTS WITH PREDICTION
        .not('alert_confidence', 'is', null),  // ✅ ONLY ALERTS WITH CONFIDENCE
    ]);

    return {
      summary: {
        totalDevices,
        onlineDevices,
        alertsToday: alertsTodayRes.count ?? 0,
        criticalAlerts: criticalRes.count ?? 0,
      },
    };
  };

  const fetchRecentAlerts = async (nodeIds: string[], homes: HomeInfo[], edgeNodesData: EdgeNodeRow[]): Promise<Partial<{ recentAlerts: ReturnType<typeof adaptAlerts> }>> => {
    console.log('[Dashboard] fetchRecentAlerts - nodeIds:', nodeIds);
    
    // If no nodes exist, return empty alerts
    if (nodeIds.length === 0) {
      console.log('[Dashboard] No nodes in fleet - returning empty alerts');
      return { recentAlerts: [] };
    }

    // CRITICAL FIX: Only fetch alerts from last 24 hours AND from nodes in the fleet
    const last24Hours = new Date(Date.now() - 24 * 3600 * 1000).toISOString();
    console.log('[Dashboard] Querying alerts since:', last24Hours);
    
    const { data, error } = await supabase
      .from('alerts_all' as any)
      .select(
        [
          'alert_id',
          'alert_first_seen_at',
          'alert_source_ip',
          'alert_target_ip',
          'alert_predicted_label',
          'alert_confidence',
          'alert_severity',
          'alert_status',
          'alert_threat_type',
          'alert_node_id',
          'alert_home_id',
          'alert_source_text',
          'alert_sequence_values_json',
          'alert_sequence_steps_json',
        ].join(', '),
      )
      .gte('alert_first_seen_at', last24Hours)  // ✅ ONLY LAST 24 HOURS
      .in('alert_node_id', nodeIds)  // ✅ ONLY ALERTS FROM FLEET NODES
      .not('alert_predicted_label', 'is', null)  // ✅ ONLY ALERTS WITH PREDICTION (model processed)
      .not('alert_confidence', 'is', null)  // ✅ ONLY ALERTS WITH CONFIDENCE SCORE
      .order('alert_first_seen_at', { ascending: false })
      .limit(20);

    console.log('[Dashboard] Alerts query result:', { data, error, count: data?.length });

    if (error) {
      console.error('[Dashboard] Alerts query error:', error);
      return {};
    }
    
    // Enrich alert rows with home/node names for user-friendly display
    const enrichedRows = ((data ?? []) as any[]).map((row: any) => {
      const home = homes.find(h => h.id === row.alert_home_id);
      const node = edgeNodesData.find((n: any) => n.id === row.alert_node_id);
      return {
        ...row,
        home_name: home?.name ?? '',
        home_code: home?.code ?? '',
        node_code: (node as any)?.node_code ?? '',
        node_location: (node as any)?.location_text ?? '',
      };
    });
    
    const adapted = adaptAlerts(enrichedRows as unknown as AlertRow[]);
    console.log('[Dashboard] Adapted alerts:', adapted);
    
    // Group DoS/DDoS alerts by source IP
    const aggregated = aggregateAlerts(adapted);
    console.log('[Dashboard] Aggregated alerts:', aggregated);
    
    // Convert AggregatedAlert back to EnhancedAlertUIModel (which extends AlertUIModel)
    const finalAlerts = aggregated.map(agg => ({
      ...agg.alerts[0],
      isAggregated: agg.isGroup,
      aggregatedCount: agg.count,
      aggregatedAlerts: agg.alerts,
    }));
    
    return { recentAlerts: finalAlerts as unknown as AlertUIModel[] };
  };

  const fetchHomes = async (): Promise<Partial<{ homes: HomeInfo[] }>> => {
    const { data, error } = await supabase.from('homes' as any).select('id, code, name').order('code');
    if (error) return {};
    return { homes: (data ?? []) as unknown as HomeInfo[] };
  };

  const fetchTrafficSeries = async (homes: HomeInfo[], nodeIds: string[]): Promise<Partial<{ trafficSeries: TrafficSeriesPoint[] }>> => {
    const since = new Date(Date.now() - 24 * 3_600_000).toISOString();
    
    // If no nodes exist, return empty traffic series
    if (nodeIds.length === 0) {
      const hourMap: Record<string, TrafficSeriesPoint> = {};
      for (let h = 0; h < 24; h++) {
        const key = `${String(h).padStart(2, '0')}:00`;
        const pt: TrafficSeriesPoint = { hour: key };
        homes.forEach((home) => {
          pt[home.code] = 0;
        });
        hourMap[key] = pt;
      }
      return { trafficSeries: Object.values(hourMap) };
    }

    const { data, error } = await supabase
      .from('alerts_all' as any)
      .select('alert_first_seen_at, alert_home_id')
      .gte('alert_first_seen_at', since)
      .in('alert_node_id', nodeIds)  // ✅ ONLY ALERTS FROM FLEET NODES
      .not('alert_predicted_label', 'is', null)  // ✅ ONLY ALERTS WITH PREDICTION
      .not('alert_confidence', 'is', null)  // ✅ ONLY ALERTS WITH CONFIDENCE
      .order('alert_first_seen_at', { ascending: true });

    if (error || !data) return {};

    const hourMap: Record<string, TrafficSeriesPoint> = {};
    for (let h = 0; h < 24; h++) {
      const key = `${String(h).padStart(2, '0')}:00`;
      const pt: TrafficSeriesPoint = { hour: key };
      homes.forEach((home) => {
        pt[home.code] = 0;
      });
      hourMap[key] = pt;
    }

    (data as any[]).forEach((row: any) => {
      const h = new Date(row.alert_first_seen_at as string).getHours();
      const key = `${String(h).padStart(2, '0')}:00`;
      const homeInfo = homes.find((hm) => hm.id === row.alert_home_id);
      if (homeInfo && hourMap[key]) {
        hourMap[key][homeInfo.code] = (hourMap[key][homeInfo.code] as number) + 1;
      }
    });

    return { trafficSeries: Object.values(hourMap) };
  };

  const fetchEdgeNodes = async (): Promise<Partial<{ edgeNodes: EdgeNodeRow[] }>> => {
    const { data, error } = await supabase
      .from('edge_nodes' as any)
      .select(
        [
          'id',
          'node_code',
          'status',
          'location_text',
          'ip_address',
          'last_seen_at',
          'current_cpu_percent',
          'current_ram_percent',
          'current_temperature_c',
          'current_latency_ms',
          'framework',
          'model_version_text',
          'home_id',
        ].join(', '),
      )
      .order('node_code', { ascending: true });

    if (error) {
      console.error('[Dashboard] Error fetching edge nodes:', error);
      return {};
    }
    
    console.log('[Dashboard] Fetched edge nodes:', data?.length ?? 0);
    return { edgeNodes: (data ?? []) as unknown as EdgeNodeRow[] };
  };

  const fetchAttackDist = async (nodeIds: string[]): Promise<Partial<{ attackDist: AttackDistPoint[] }>> => {
    const since = new Date(Date.now() - 24 * 3_600_000).toISOString();
    
    // If no nodes exist, return empty attack distribution
    if (nodeIds.length === 0) {
      return { attackDist: [] };
    }

    const { data, error } = await supabase
      .from('alerts_all' as any)
      .select('alert_threat_type')
      .gte('alert_first_seen_at', since)
      .in('alert_node_id', nodeIds)  // ✅ ONLY ALERTS FROM FLEET NODES
      .not('alert_predicted_label', 'is', null)  // ✅ ONLY ALERTS WITH PREDICTION
      .not('alert_confidence', 'is', null);  // ✅ ONLY ALERTS WITH CONFIDENCE

    if (error || !data) return {};

    const countMap: Record<string, number> = {};
    (data as any[]).forEach((row: any) => {
      const t = (row.alert_threat_type as string) ?? 'Unknown';
      countMap[t] = (countMap[t] ?? 0) + 1;
    });

    const attackDist: AttackDistPoint[] = Object.entries(countMap)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([name, value]) => ({ name, value, color: getAttackColor(name) }));

    return { attackDist };
  };

  const fetchModelStats = async (): Promise<Partial<{ modelStats: ModelStats }>> => {
    const { data, error } = await supabase
      .from('model_versions' as any)
      .select(
        'version, status, accuracy, f1_score, precision, recall, latency_ms, throughput_per_s, false_positive_rate',
      )
      .in('status', ['deployed', 'active'])
      .order('created_at', { ascending: false })
      .limit(1)
      .single();

    if (error || !data) return {};
    const row = data as any;
    return {
      modelStats: {
        version: row.version,
        status: row.status,
        accuracy: row.accuracy,
        f1_score: row.f1_score,
        precision: row.precision,
        recall: row.recall,
        latency_ms: row.latency_ms,
        throughput_per_s: row.throughput_per_s,
        false_positive_rate: row.false_positive_rate,
      },
    };
  };

  const fetchNetworkFlowMetrics = async (): Promise<Partial<{ networkFlowMetrics: any[] }>> => {
    console.log('[Dashboard] fetchNetworkFlowMetrics - START');
    
    const since = new Date(Date.now() - 24 * 3_600_000).toISOString();
    console.log('[Dashboard] Querying network flows since:', since);
    
    // STRATEGY 1: Try with flow_ts filter (most common case)
    const { data: flowData, error: flowError, count } = await supabase
      .from('network_flows_feedback_all' as any)
      .select('*', { count: 'exact' })
      .gte('flow_ts', since)
      .order('flow_ts', { ascending: true });

    console.log('[Dashboard] Strategy 1 (flow_ts filter) - count:', count, 'data length:', flowData?.length, 'error:', flowError);

    // If we got data, process it
    if (!flowError && flowData && flowData.length > 0) {
      return processFlowData(flowData as any[]);
    }

    // STRATEGY 2: Try with flow_created_at filter (alternative timestamp field)
    console.log('[Dashboard] Strategy 2: Trying flow_created_at filter...');
    const { data: createdData, error: createdError } = await supabase
      .from('network_flows_feedback_all' as any)
      .select('*')
      .gte('flow_created_at', since)
      .order('flow_created_at', { ascending: true });
    
    console.log('[Dashboard] Strategy 2 result:', createdData?.length, 'error:', createdError);
    
    if (!createdError && createdData && createdData.length > 0) {
      return processFlowData(createdData as any[]);
    }

    // STRATEGY 3: Get ALL data without time filter (user said there's LOTS of data)
    console.log('[Dashboard] Strategy 3: Fetching ALL data (no time filter)...');
    const { data: allData, error: allError } = await supabase
      .from('network_flows_feedback_all' as any)
      .select('*')
      .limit(1000)  // Limit to prevent overwhelming the browser
      .order('flow_id', { ascending: false });  // Get most recent by ID
    
    console.log('[Dashboard] Strategy 3 result:', allData?.length, 'error:', allError);
    
    if (!allError && allData && allData.length > 0) {
      console.log('[Dashboard] SUCCESS! Found data without time filter. Sample:', allData[0]);
      return processFlowData(allData as any[]);
    }

    // STRATEGY 4: Last resort - try with minimal select
    console.log('[Dashboard] Strategy 4: Minimal select...');
    const { data: minimalData, error: minimalError } = await supabase
      .from('network_flows_feedback_all' as any)
      .select('flow_id, flow_ts, flow_created_at, is_anomaly, flow_total_bytes')
      .limit(1000);
    
    console.log('[Dashboard] Strategy 4 result:', minimalData?.length, 'error:', minimalError);
    
    if (!minimalError && minimalData && minimalData.length > 0) {
      return processFlowData(minimalData as any[]);
    }

    console.error('[Dashboard] All strategies failed. Last error:', minimalError || allError || createdError || flowError);
    return { networkFlowMetrics: [] };
  };

  // Helper function to process flow data
  function processFlowData(flowData: any[]) {
    console.log('[Dashboard] Processing', flowData.length, 'flow records');
    console.log('[Dashboard] Sample record:', flowData[0]);
    console.log('[Dashboard] Sample record keys:', Object.keys(flowData[0] || {}));

    // Group by hour - use current time as reference for "last 24h"
    const now = new Date();
    const hourMap: Record<string, { total: number; anomalies: number; bytes: number }> = {};
    
    for (let h = 0; h < 24; h++) {
      const key = `${String(h).padStart(2, '0')}:00`;
      hourMap[key] = { total: 0, anomalies: 0, bytes: 0 };
    }

    let processedCount = 0;
    let anomalyCount = 0;
    let errorCount = 0;
    let nullTimestampCount = 0;

    flowData.forEach((row: any, idx: number) => {
      try {
        // Handle different timestamp field names
        const timestamp = row.flow_ts || row.flow_created_at || row.created_at;
        
        // If no timestamp, distribute evenly across hours (fallback strategy)
        if (!timestamp) {
          nullTimestampCount++;
          // Use index to distribute across hours
          const h = idx % 24;
          const key = `${String(h).padStart(2, '0')}:00`;
          
          hourMap[key].total += 1;
          processedCount += 1;
          
          // Check for anomaly
          const isAnomaly = row.is_anomaly === true || 
                           row.is_anomaly === 1 || 
                           row.is_anomaly === '1' ||
                           row.is_anomaly === 'true';
          
          if (isAnomaly) {
            hourMap[key].anomalies += 1;
            anomalyCount += 1;
          }
          
          // Handle bytes
          const bytes = row.flow_total_bytes || row.total_bytes || row.bytes || 0;
          hourMap[key].bytes += Number(bytes) || 0;
          
          return;
        }

        // Normal case: use timestamp
        const h = new Date(timestamp).getHours();
        const key = `${String(h).padStart(2, '0')}:00`;
        
        if (!hourMap[key]) {
          hourMap[key] = { total: 0, anomalies: 0, bytes: 0 };
        }
        
        hourMap[key].total += 1;
        processedCount += 1;
        
        // Check for anomaly (handle different field names and types)
        const isAnomaly = row.is_anomaly === true || 
                         row.is_anomaly === 1 || 
                         row.is_anomaly === '1' ||
                         row.is_anomaly === 'true';
        
        if (isAnomaly) {
          hourMap[key].anomalies += 1;
          anomalyCount += 1;
        }
        
        // Handle bytes (different field names)
        const bytes = row.flow_total_bytes || row.total_bytes || row.bytes || 0;
        hourMap[key].bytes += Number(bytes) || 0;
      } catch (e) {
        errorCount++;
        console.error('[Dashboard] Error processing row:', e, row);
      }
    });

    console.log('[Dashboard] Processed:', processedCount, 'records');
    console.log('[Dashboard] Anomalies:', anomalyCount);
    console.log('[Dashboard] NULL timestamps:', nullTimestampCount);
    console.log('[Dashboard] Errors:', errorCount);

    const metrics = Object.entries(hourMap).map(([hour, stats]) => ({
      hour,
      flows: stats.total,
      anomalies: stats.anomalies,
      bytes: Math.round(stats.bytes / (1024 * 1024)), // Convert to MB
    }));

    const totalFlows = metrics.reduce((sum, m) => sum + m.flows, 0);
    console.log('[Dashboard] Final metrics - Total flows:', totalFlows);
    console.log('[Dashboard] Metrics array:', metrics);

    return { networkFlowMetrics: metrics };
  }

  const loadAll = useCallback(async () => {
    const uid = userIdRef.current;
    if (!uid) return;

    setIsLoading(true);
    setError(null);

    // Fetch edge nodes AND homes first — needed for enriching alerts with user-friendly names
    const [edgeNodesResult, homesResult] = await Promise.all([
      fetchEdgeNodes(),
      fetchHomes(),
    ]);
    
    const edgeNodesData = edgeNodesResult.edgeNodes ?? [];
    const homes = homesResult.homes ?? [];
    const nodeIds = edgeNodesData.map((n: any) => n.id);

    console.log('[Dashboard] Node IDs from fleet:', nodeIds);
    console.log('[Dashboard] Number of nodes:', nodeIds.length);

    const results = await Promise.allSettled([
      fetchSummary(nodeIds),
      fetchRecentAlerts(nodeIds, homes, edgeNodesData),
      fetchTrafficSeries(homes, nodeIds),
      fetchAttackDist(nodeIds),
      fetchModelStats(),
      fetchNetworkFlowMetrics(),
    ]);

    const merged: Record<string, unknown> = { homes, edgeNodes: edgeNodesData };
    results.forEach((r, idx) => {
      if (r.status === 'fulfilled') {
        Object.assign(merged, r.value);
        console.log(`[Dashboard] Result ${idx}:`, r.value);
      } else {
        console.error(`[Dashboard] Result ${idx} failed:`, r.reason);
      }
    });

    console.log('[Dashboard] Merged data:', merged);

    writeCache(uid, merged as Partial<Omit<DashboardCacheState, 'fetchedAt'>>);
    syncFromCache(uid);

    setIsLoading(false);
  }, [setIsLoading, setError, syncFromCache]);

  const subscribeRealtime = useCallback(() => {
    const uid = userIdRef.current;
    if (!uid) return;
    if (channelRef.current) return;

    channelRef.current = supabase
      .channel(`dashboard-realtime-${uid}`)
      .on(
        'postgres_changes' as any,
        { event: '*', schema: 'public', table: 'alerts_all' },
        async (payload: any) => {
          const current = userIdRef.current;
          if (!current) return;
          
          // Only play sound for INSERT events (new alerts)
          if (payload.eventType === 'INSERT') {
            const newAlert = payload.new as any;
            const newSourceIp = newAlert?.alert_source_ip || null;
            
            // Only play sound if source IP is different from last alert
            if (newSourceIp !== lastAlertSourceIpRef.current) {
              console.log(`[Dashboard] New alert from different source IP: ${newSourceIp} (previous: ${lastAlertSourceIpRef.current})`);
              playSound();
              lastAlertSourceIpRef.current = newSourceIp;
            } else {
              console.log(`[Dashboard] Alert from same source IP (${newSourceIp}) - sound suppressed`);
            }
          }
          
          invalidateCache(current);
          void loadAll();
        },
      )
      .on(
        'postgres_changes' as any,
        { event: '*', schema: 'public', table: 'edge_nodes' },
        () => {
          const current = userIdRef.current;
          if (!current) return;
          invalidateCache(current);
          void loadAll();
        },
      )
      .subscribe();
  }, [loadAll, playSound]);

  useEffect(() => {
    if (!userId) {
      setIsLoading(false);
      return;
    }

    if (isCacheValid(userId)) {
      syncFromCache(userId);
      setIsLoading(false);
    } else {
      void loadAll();
    }

    subscribeRealtime();

    // CRITICAL FIX: Auto-refresh every 5 seconds (down from 30s) to match Live Monitor's 2s polling
    // This ensures Dashboard shows near-realtime data consistent with Live Monitor
    const refreshInterval = setInterval(() => {
      const current = userIdRef.current;
      if (current) {
        invalidateCache(current);
        void loadAll();
      }
    }, 5000); // 5 seconds (compromise between realtime and performance)

    return () => {
      if (channelRef.current) {
        supabase.removeChannel(channelRef.current);
        channelRef.current = null;
      }
      clearInterval(refreshInterval);
    };
  }, [userId, loadAll, subscribeRealtime, syncFromCache, setIsLoading]);

  /**
   * Enable/Isolate node by publishing to IoT Core topic 'Ddos'
   * @param ipAddress - IP address of the node
   * @param action - 'unblock' for DDoS (enable node) or 'isolate' for other attacks
   */
  const enableNode = useCallback(async (ipAddress: string, action: 'unblock' | 'isolate') => {
    try {
      console.log(`[Dashboard] ${action === 'unblock' ? 'Enabling' : 'Isolating'} node - IP:`, ipAddress, 'Action:', action);
      
      // Try API Gateway endpoint (Lambda will handle IoT Core publishing)
      const API_BASE_URL = import.meta.env.VITE_API_GATEWAY_URL || 'https://fbujw415e6.execute-api.ap-southeast-2.amazonaws.com/prod';
      
      const response = await fetch(`${API_BASE_URL}/unblock-ddos`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ip_address: ipAddress, action }),
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('[Dashboard] Node action via API Gateway - success:', result);
        return { success: true };
      }
      
      // If API Gateway fails, return error
      const error = await response.json().catch(() => ({ error: 'Unknown error' }));
      console.error('[Dashboard] API Gateway error:', error);
      return { success: false, error: error.error || `HTTP ${response.status}` };
      
    } catch (error) {
      console.error('[Dashboard] Error with node action:', error);
      return { success: false, error: String(error) };
    }
  }, []);

  /**
   * Dismiss alert (delete from alerts_all table)
   */
  const dismissAlert = useCallback(async (alertId: string) => {
    try {
      console.log('[Dashboard] Dismissing alert:', alertId);
      
      const { error } = await supabase
        .from('alerts_all' as any)
        .delete()
        .eq('alert_id', alertId);
      
      if (error) {
        console.error('[Dashboard] Error dismissing alert:', error);
        return { success: false, error: error.message };
      }
      
      console.log('✅ Alert dismissed successfully');
      
      // Refresh dashboard data
      const current = userIdRef.current;
      if (current) {
        invalidateCache(current);
        void loadAll();
      }
      
      return { success: true };
    } catch (error) {
      console.error('[Dashboard] Error dismissing alert:', error);
      return { success: false, error: String(error) };
    }
  }, [loadAll]);

  return { ...store, refresh: loadAll, enableNode, dismissAlert };
}
