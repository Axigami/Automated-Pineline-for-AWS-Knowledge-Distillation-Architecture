import { useCallback, useEffect, useRef, useMemo, useState } from 'react';
import { supabase } from '../../../core/lib/supabaseClient';
import { useThreatAnalyticsStore } from '../model/store';
import {
  adaptFlow,
  adaptFlows,
  buildAggregation,
  buildTopAttackers,
  buildTimeline,
} from '../model/adapter';
import type { FlowRow, LabelFeedbackRequest } from '../model/types';

// ─── @ Syntax Parser ──────────────────────────────────────────────────────────
// Cú pháp hỗ trợ:
//   DDoS           → filter label (all homes)
//   @HomeName      → filter home by name (all labels)
//   DDoS @HomeName → filter label + home
//
// SQL-injection safety: Supabase SDK dùng parameterized queries cho
// .ilike() .eq() .in() — không bao giờ nối chuỗi vào raw SQL.
// Thêm: validate home bằng regex, giới hạn độ dài.

const HOME_TOKEN_REGEX = /^[\w\-\.\s]+$/; // cho phép ký tự an toàn (bao gồm space trong tên)
const MAX_INPUT_LEN = 200;

interface ParsedAtQuery {
  label: string;    // free-text label ('' = no filter)
  homeName: string; // home name sau @ ('' = no filter)
}

/**
 * Parse cú pháp @ từ query string.
 * Nếu truyền knownHomeNames, sẽ thử match tên home có dấu cách (e.g. @Smart Office B).
 * Ưu tiên: quoted (@"...") → known name (longest match) → simple word.
 */
export function parseAtQuery(raw: string, knownHomeNames: string[] = []): ParsedAtQuery {
  const trimmed = raw.slice(0, MAX_INPUT_LEN).trim();

  // 1) Quoted syntax: @"Smart Office B"
  const quotedMatch = trimmed.match(/@"([^"]+)"/);
  if (quotedMatch) {
    const labelPart = trimmed.replace(/@"[^"]+"/g, '').trim();
    return { label: labelPart, homeName: quotedMatch[1] };
  }

  // 2) Thử match tên home đã biết (longest match first) sau dấu @
  //    Ví dụ: "DDoS @Smart Office B" → homeName = "Smart Office B", label = "DDoS"
  if (knownHomeNames.length > 0) {
    // Sort dài → ngắn để ưu tiên tên dài nhất (tránh match "Smart" thay vì "Smart Office B")
    const sorted = [...knownHomeNames].sort((a, b) => b.length - a.length);
    for (const name of sorted) {
      // Tìm @name trong query (case-insensitive)
      const atIdx = trimmed.toLowerCase().indexOf('@' + name.toLowerCase());
      if (atIdx >= 0) {
        const before = trimmed.slice(0, atIdx).trim();
        const after  = trimmed.slice(atIdx + 1 + name.length).trim();
        const labelPart = (before + ' ' + after).trim();
        return { label: labelPart, homeName: name };
      }
    }
  }

  // 3) Fallback: simple @word (tên không có dấu cách)
  const simpleMatch = trimmed.match(/@([\w\-\.]+)/);
  const homeName = simpleMatch ? simpleMatch[1] : '';
  const labelPart = trimmed.replace(/@[\w\-\.]+/g, '').trim();

  return { label: labelPart, homeName };
}

// ─── Controller Hook ─────────────────────────────────────────────────────────
export function useThreatAnalytics() {
  const store = useThreatAnalyticsStore();

  // ─── Home name → ID mapping (fetched from homes table) ──────────────────
  const [homeNameToId, setHomeNameToId] = useState<Record<string, string>>({});
  const [homeIdToName, setHomeIdToName] = useState<Record<string, string>>({});

  // Ref trỏ tới store mới nhất – tránh stale closure
  const storeRef = useRef(store);
  storeRef.current = store;

  // Ref giữ Supabase Realtime channel để cleanup khi unmount
  const channelRef = useRef<ReturnType<typeof supabase.channel> | null>(null);

  // Ref cho home maps
  const homeNameToIdRef = useRef(homeNameToId);
  homeNameToIdRef.current = homeNameToId;
  const homeIdToNameRef = useRef(homeIdToName);
  homeIdToNameRef.current = homeIdToName;

  // Fetch danh sách homes để build mapping name ↔ id
  useEffect(() => {
    (async () => {
      const { data } = await supabase
        .from('homes' as any)
        .select('id, name');
      if (data && Array.isArray(data)) {
        const n2i: Record<string, string> = {};
        const i2n: Record<string, string> = {};
        data.forEach((h: any) => {
          if (h.name && h.id) {
            n2i[h.name.toLowerCase()] = h.id;
            i2n[h.id] = h.name;
          }
        });
        setHomeNameToId(n2i);
        setHomeIdToName(i2n);
      }
    })();
  }, []);

  const queryLogs = useCallback(async () => {
    const { queryParams } = storeRef.current;
    storeRef.current.setIsLoading(true);
    storeRef.current.setError(null);

    // Parse cú pháp @ từ query string (truyền danh sách tên home đã biết để match multi-word)
    const knownNames = Object.values(homeIdToNameRef.current) as string[];
    const parsed = parseAtQuery(queryParams.query, knownNames);

    let q = supabase
      .from('network_flows_feedback_all' as any)
      .select(
        'flow_id, flow_home_id, flow_node_id, flow_ts, flow_protocol, ' +
        'flow_src_ip, flow_dst_ip, flow_src_port, flow_dst_port, ' +
        'flow_duration_s, flow_in_bytes, flow_out_bytes, flow_tcp_flags, flow_total_bytes, ' +
        'predicted_label, confidence, anomaly_score, is_anomaly, inference_logic, ' +
        'feedback_action, feedback_true_label, feedback_note, feedback_user_id, feedback_created_at'
      )
      .order('flow_ts', { ascending: false })
      .limit(500);

    // ── Filter thời gian ──────────────────────────────────────────────────────
    if (queryParams.from && queryParams.to) {
      q = (q as any)
        .gte('flow_ts', queryParams.from)
        .lte('flow_ts', queryParams.to);
    }

    // ── Filter label (ilike – parameterized → safe) ───────────────────────────
    if (parsed.label) {
      q = (q as any).ilike('predicted_label', `%${parsed.label}%`);
    }

    // ── Filter home by name → resolve to ID (eq – parameterized → safe) ──────
    if (parsed.homeName) {
      const resolvedId = homeNameToIdRef.current[parsed.homeName.toLowerCase()];
      if (resolvedId) {
        q = (q as any).eq('flow_home_id', resolvedId);
      } else {
        // Không tìm thấy home name → trả empty
        storeRef.current.setFlows([]);
        storeRef.current.setAggregation([]);
        storeRef.current.setTopAttackers([]);
        storeRef.current.setTimeline([]);
        storeRef.current.setError(`Không tìm thấy home "${parsed.homeName}"`);
        storeRef.current.setIsLoading(false);
        return;
      }
    }

    const { data, error } = await q;

    if (error) {
      storeRef.current.setError('Truy vấn log thất bại: ' + error.message);
      storeRef.current.setIsLoading(false);
      return;
    }

    const flows = adaptFlows((data ?? []) as unknown as FlowRow[]);

    storeRef.current.setFlows(flows);
    storeRef.current.setAggregation(buildAggregation(flows));
    storeRef.current.setTopAttackers(buildTopAttackers(flows));
    storeRef.current.setTimeline(buildTimeline(flows));
    storeRef.current.setIsLoading(false);
  }, []);

  // Fetch lần đầu khi mount + subscribe realtime
  useEffect(() => {
    queryLogs();

    // ── Supabase Realtime: lắng nghe INSERT mới trên network_flows_feedback_all ──
    // Pattern giống fleet-mgmt: patch store cục bộ thay vì refetch toàn bộ
    const channel = supabase
      .channel('threat_flows_live')
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'network_flows_feedback_all',
        },
        (payload) => {
          // Adapt raw DB row → UI model rồi prepend vào store
          const newFlow = adaptFlow(payload.new as FlowRow);
          storeRef.current.prependFlow(newFlow);

          // IMPORTANT: sau set() Zustand đã update nội bộ ngay lập tức.
          // Dùng getState() để lấy flows mới nhất (storeRef.current.flows vẫn
          // là React snapshot cũ – chưa re-render → sẽ thiếu row vừa prepend).
          const updatedFlows = useThreatAnalyticsStore.getState().flows;
          storeRef.current.setAggregation(buildAggregation(updatedFlows));
          storeRef.current.setTopAttackers(buildTopAttackers(updatedFlows));
          storeRef.current.setTimeline(buildTimeline(updatedFlows));
        }
      )
      .subscribe();

    channelRef.current = channel;

    // Cleanup: unsubscribe khi component unmount
    return () => {
      if (channelRef.current) {
        supabase.removeChannel(channelRef.current);
        channelRef.current = null;
      }
    };
  }, [queryLogs]);

  /**
   * FR 2.2 – Human-in-the-loop Feedback
   * Optimistic update: cập nhật UI tức thì, sau đó ghi DB.
   */
  const submitFeedback = useCallback(
    async (req: LabelFeedbackRequest) => {
      storeRef.current.updateFlowLabel(req.flowId, req.trueLabel);

      const { error } = await supabase
        .from('network_flows_feedback_all' as any)
        .update({
          feedback_true_label: req.trueLabel,
          feedback_action: 'relabel',
          feedback_note: req.note ?? null,
          feedback_created_at: new Date().toISOString(),
        })
        .eq('flow_id', req.flowId);

      if (error) {
        storeRef.current.setError('Không thể gán nhãn lại: ' + error.message);
      } else {
        storeRef.current.setFeedbackSuccess(
          `Đã cập nhật nhãn cho flow ${req.flowId.slice(0, 8)}…`
        );
        setTimeout(() => storeRef.current.setFeedbackSuccess(null), 3000);
        const updatedFlows = storeRef.current.flows;
        storeRef.current.setAggregation(buildAggregation(updatedFlows));
        storeRef.current.setTopAttackers(buildTopAttackers(updatedFlows));
        storeRef.current.setTimeline(buildTimeline(updatedFlows));
      }
    },
    []
  );

  // Danh sách unique home names: lấy flow_home_id từ flows đã load → map về bảng homes lấy tên
  const availableHomes = useMemo(() => {
    const seen = new Set<string>();
    const names: string[] = [];
    store.flows.forEach((f) => {
      if (f.homeId && !seen.has(f.homeId)) {
        seen.add(f.homeId);
        const name = homeIdToName[f.homeId];
        if (name) names.push(name);
      }
    });
    return names.sort();
  }, [store.flows, homeIdToName]);

  return {
    ...store,
    queryLogs,
    submitFeedback,
    availableHomes,
  };
}
