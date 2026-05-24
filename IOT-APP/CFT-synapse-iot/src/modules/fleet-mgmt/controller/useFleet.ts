import { useEffect, useCallback, useRef } from 'react';
import { supabase } from '../../../core/lib/supabaseClient';
import { useFleetStore } from '../model/store';
import { adaptNodes, adaptNode } from '../model/adapter';
import type { EdgeNodeRow, NodeTelemetryRow, TelemetryPoint } from '../model/types';

/**
 * useFleet – Controller hook.
 * Optimized for high-frequency real-time updates.
 */
export function useFleet() {
  const store = useFleetStore();

  // Ref always points to the latest store to avoid closure staleness without re-triggering effects
  const storeRef = useRef(store);
  storeRef.current = store;

  const channelRef = useRef<ReturnType<typeof supabase.channel> | null>(null);

  const fetchTelemetry = useCallback(async (nodeIds: string[]) => {
    if (!nodeIds.length) return;

    const { data, error } = await supabase
      .from('node_telemetry')
      .select('id, node_id, ts, cpu_percent, ram_percent, temperature_c, latency_ms')
      .in('node_id', nodeIds)
      .order('ts', { ascending: false })
      .limit(nodeIds.length * 20);

    if (error || !data) return;

    const map: Record<string, TelemetryPoint[]> = {};
    for (const row of data as NodeTelemetryRow[]) {
      if (!map[row.node_id]) map[row.node_id] = [];
      if (map[row.node_id].length < 20) {
        map[row.node_id].push({
          ts: row.ts,
          cpu: row.cpu_percent ?? 0,
          ram: row.ram_percent ?? 0,
          temp: row.temperature_c ?? 0,
          latency: row.latency_ms ?? 0,
        });
      }
    }
    Object.keys(map).forEach(k => map[k].reverse());

    storeRef.current.setTelemetryMap(map);
  }, []);

  const fetchNodes = useCallback(async () => {
    storeRef.current.setIsLoading(true);

    const { data, error } = await supabase
      .from('edge_nodes')
      .select(
        'id, home_id, node_code, status, location_text, ip_address, last_seen_at, ' +
        'current_cpu_percent, current_ram_percent, current_temperature_c, current_latency_ms, ' +
        'model_version_text, framework, deployed_model_version_id'
      )
      .order('status', { ascending: true });

    if (error) {
      storeRef.current.setError('Không thể tải danh sách thiết bị');
    } else {
      const rows = (data ?? []) as unknown as EdgeNodeRow[];
      storeRef.current.setNodes(adaptNodes(rows));
      storeRef.current.setError(null);

      const ids = rows.map(r => r.id);
      fetchTelemetry(ids);
    }

    storeRef.current.setIsLoading(false);
  }, [fetchTelemetry]);

  useEffect(() => {
    fetchNodes();

    // Subscribe to real-time changes
    const channel = supabase
      .channel('edge_nodes_fleet_live')
      .on(
        'postgres_changes',
        { event: 'UPDATE', schema: 'public', table: 'edge_nodes' },
        (payload) => {
          // OPTIMIZATION: Patch single node state instead of re-fetching everything
          const adapted = adaptNode(payload.new as EdgeNodeRow);
          storeRef.current.updateNode(adapted.id, adapted);
        }
      )
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'node_telemetry' },
        (payload) => {
          // OPTIMIZATION: Append single telemetry point instead of re-fetching history
          const row = payload.new as NodeTelemetryRow;
          const point: TelemetryPoint = {
            ts: row.ts,
            cpu: row.cpu_percent ?? 0,
            ram: row.ram_percent ?? 0,
            temp: row.temperature_c ?? 0,
            latency: row.latency_ms ?? 0,
          };
          storeRef.current.appendTelemetryPoint(row.node_id, point);
        }
      )
      .subscribe();

    channelRef.current = channel;

    return () => {
      if (channelRef.current) {
        supabase.removeChannel(channelRef.current);
        channelRef.current = null;
      }
    };
  }, [fetchNodes]); // fetchNodes is stable

  const restartNode = useCallback(async (nodeId: string): Promise<void> => {
    const { data: nodeRow } = await supabase
      .from('edge_nodes')
      .select('deployed_model_version_id')
      .eq('id', nodeId)
      .maybeSingle();

    const modelVersionId = (nodeRow as { deployed_model_version_id?: string | null } | null)?.deployed_model_version_id ?? null;
    const now = new Date().toISOString();

    const { error } = await supabase
      .from('deployments_all')
      .insert({
        deployment_id: crypto.randomUUID(),
        target_node_id: nodeId,
        deployment_model_version_id: modelVersionId,
        deployment_status: 'pending',
        deployment_created_at: now,
        audit_action: 'RESTART_NODE',
        audit_target: nodeId,
        audit_status: 'pending',
        audit_created_at: now,
      } as any);

    if (error) throw new Error(error.message);
  }, []);

  // ─── ADD NODE: Tạo Home mới trước → rồi tạo Edge Node gắn vào Home đó ─────
  const addNode = useCallback(async (payload: { nodeCode: string; location: string; ipAddress: string; framework: string }): Promise<void> => {
    const now = new Date().toISOString();
    const homeId = crypto.randomUUID();

    // 1. Tạo Home mới (đồng bộ name/region với node)
    const { error: homeErr } = await supabase
      .from('homes' as any)
      .insert({
        id: homeId,
        code: payload.nodeCode,
        name: `Home ${payload.nodeCode}`,
        region: payload.location,
        created_at: now,
        cloud_verification_confidence_threshold: 70,
        data_drift_alert_level: 0.15,
      } as any);

    if (homeErr) throw new Error(`Không thể tạo Home: ${homeErr.message}`);

    // 2. Tạo Edge Node gắn vào Home vừa tạo (id bắt buộc theo schema Database_App_IoT.sql)
    const nodeId = crypto.randomUUID();
    const { error: nodeErr } = await supabase
      .from('edge_nodes')
      .insert({
        id: nodeId,
        home_id: homeId,
        node_code: payload.nodeCode,
        location_text: payload.location,
        ip_address: payload.ipAddress,
        framework: payload.framework,
        status: 'online',
        current_cpu_percent: 0,
        current_ram_percent: 0,
        current_temperature_c: 0,
        current_latency_ms: 0,
        created_at: now,
      } as any);

    if (nodeErr) throw new Error(`Không thể tạo Edge Node: ${nodeErr.message}`);
    fetchNodes();
  }, [fetchNodes]);

  // ─── UPDATE NODE LOCATION: Đồng bộ location_text → homes.region ───────────
  const updateNodeLocation = useCallback(async (nodeId: string, newLocation: string): Promise<void> => {
    // Lấy home_id của node
    const { data: nodeData } = await supabase
      .from('edge_nodes')
      .select('home_id')
      .eq('id', nodeId)
      .single();

    if (!nodeData) throw new Error('Không tìm thấy Edge Node');

    // Cập nhật edge_nodes.location_text
    const { error: nodeErr } = await supabase
      .from('edge_nodes')
      .update({ location_text: newLocation } as any)
      .eq('id', nodeId);

    if (nodeErr) throw new Error(nodeErr.message);

    // Đồng bộ ngược → homes.region
    const { error: homeErr } = await supabase
      .from('homes' as any)
      .update({ region: newLocation } as any)
      .eq('id', (nodeData as any).home_id);

    if (homeErr) console.warn('Không đồng bộ được Home.region:', homeErr.message);

    fetchNodes();
  }, [fetchNodes]);

  // ─── DELETE NODE: Cascade delete Edge Node + related records + Home ───────
  const deleteNode = useCallback(async (nodeId: string): Promise<void> => {
    // Get home_id before deleting node
    const { data: nodeData } = await supabase
      .from('edge_nodes')
      .select('home_id')
      .eq('id', nodeId)
      .single();

    // CASCADE DELETE: Delete all related records first (in correct order)
    
    // 1. Delete node_telemetry (telemetry data for this node)
    const { error: telemetryErr } = await supabase
      .from('node_telemetry')
      .delete()
      .eq('node_id', nodeId);
    
    if (telemetryErr) console.warn('Failed to delete node_telemetry:', telemetryErr.message);

    // 2. Delete network_flows_feedback_all (network flows from this node)
    const { error: flowsErr } = await supabase
      .from('network_flows_feedback_all')
      .delete()
      .eq('flow_node_id', nodeId);
    
    if (flowsErr) console.warn('Failed to delete network_flows:', flowsErr.message);

    // 3. Delete alerts_all (alerts from this node)
    const { error: alertsErr } = await supabase
      .from('alerts_all')
      .delete()
      .eq('alert_node_id', nodeId);
    
    if (alertsErr) console.warn('Failed to delete alerts:', alertsErr.message);

    // 4. Delete deployments_all (deployment targets for this node)
    const { error: deploymentsErr } = await supabase
      .from('deployments_all')
      .delete()
      .eq('target_node_id', nodeId);
    
    if (deploymentsErr) console.warn('Failed to delete deployments:', deploymentsErr.message);

    // 5. Now delete the Edge Node itself
    const { error: nodeErr } = await supabase
      .from('edge_nodes')
      .delete()
      .eq('id', nodeId);

    if (nodeErr) throw new Error(`Cannot delete Edge Node: ${nodeErr.message}`);

    // 6. Delete Home if no other nodes use it
    if (nodeData) {
      const { count } = await supabase
        .from('edge_nodes')
        .select('id', { count: 'exact', head: true })
        .eq('home_id', (nodeData as any).home_id);

      if (count === 0) {
        // Delete related home records before deleting home
        
        // Delete network_flows_feedback_all for this home
        await supabase
          .from('network_flows_feedback_all')
          .delete()
          .eq('flow_home_id', (nodeData as any).home_id);

        // Delete alerts_all for this home
        await supabase
          .from('alerts_all')
          .delete()
          .eq('alert_home_id', (nodeData as any).home_id);

        // Delete retrain_jobs_all for this home
        await supabase
          .from('retrain_jobs_all')
          .delete()
          .eq('job_home_id', (nodeData as any).home_id);

        // Delete users_roles_settings for this home
        await supabase
          .from('users_roles_settings')
          .delete()
          .eq('setting_home_id', (nodeData as any).home_id);

        // Finally delete the home
        await supabase
          .from('homes' as any)
          .delete()
          .eq('id', (nodeData as any).home_id);
      }
    }

    fetchNodes();
  }, [fetchNodes]);

  return {
    ...store,
    refresh: fetchNodes,
    restartNode,
    addNode,
    updateNodeLocation,
    deleteNode,
    setSelectedNodeId: store.setSelectedNodeId,
  };
}
