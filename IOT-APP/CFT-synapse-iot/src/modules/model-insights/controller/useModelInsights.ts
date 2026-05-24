import { useEffect, useCallback } from 'react';
import { supabase } from '../../../core/lib/supabaseClient';
import { useModelInsightsStore } from '../model/store';
import type { FlowInferenceRow, ModelComparisonRow } from '../model/types';

/**
 * useModelInsights – queries flow_inference + model_versions.
 */
export function useModelInsights() {
  const store = useModelInsightsStore();

  const fetchData = useCallback(async () => {
    store.setIsLoading(true);

    const [infRes, verRes] = await Promise.all([
      supabase
        .from('flow_inference')
        .select('id, flow_id, model_version_id, engine, predicted_label, confidence, anomaly_score, is_anomaly, created_at')
        .order('created_at', { ascending: false })
        .limit(200),
      supabase
        .from('model_versions')
        .select('id, version, accuracy, f1_score, precision, recall, latency_ms, false_positive_rate, created_at, model_id')
        .order('created_at', { ascending: false })
        .limit(20),
    ]);

    if (!infRes.error) store.setInferences((infRes.data ?? []) as FlowInferenceRow[]);
    if (!verRes.error) {
      // engine field từ models table (join không cần thiết, để mặc định)
      const rows: ModelComparisonRow[] = (verRes.data ?? []).map((r) => ({
        ...r,
        engine: 'cloud',
      }));
      store.setModelVersions(rows);
    }
    if (infRes.error || verRes.error) {
      store.setError('Failed to load Model Insights');
    }

    store.setIsLoading(false);
  }, [store]);

  useEffect(() => { fetchData(); }, [fetchData]);

  return { ...store, refresh: fetchData };
}
