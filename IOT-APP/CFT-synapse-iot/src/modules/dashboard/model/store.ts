import { useState } from 'react';
import type {
  AlertUIModel,
  TelemetrySummary,
  TrafficSeriesPoint,
  EdgeNodeRow,
  AttackDistPoint,
  ModelStats,
  HomeInfo,
} from './types';

/**
 * Dashboard local state hook.
 * State đầy đủ cho tất cả widget Dashboard.
 */
export function useDashboardStore() {
  const [summary,      setSummary]      = useState<TelemetrySummary | null>(null);
  const [recentAlerts, setRecentAlerts] = useState<AlertUIModel[]>([]);
  const [trafficSeries,setTrafficSeries]= useState<TrafficSeriesPoint[]>([]);
  const [homes,        setHomes]        = useState<HomeInfo[]>([]);
  const [edgeNodes,    setEdgeNodes]    = useState<EdgeNodeRow[]>([]);
  const [attackDist,   setAttackDist]   = useState<AttackDistPoint[]>([]);
  const [modelStats,   setModelStats]   = useState<ModelStats | null>(null);
  const [networkFlowMetrics, setNetworkFlowMetrics] = useState<any[]>([]);
  const [isLoading,    setIsLoading]    = useState(false);
  const [error,        setError]        = useState<string | null>(null);

  return {
    summary,      setSummary,
    recentAlerts, setRecentAlerts,
    trafficSeries,setTrafficSeries,
    homes,        setHomes,
    edgeNodes,    setEdgeNodes,
    attackDist,   setAttackDist,
    modelStats,   setModelStats,
    networkFlowMetrics, setNetworkFlowMetrics,
    isLoading,    setIsLoading,
    error,        setError,
  };
}
