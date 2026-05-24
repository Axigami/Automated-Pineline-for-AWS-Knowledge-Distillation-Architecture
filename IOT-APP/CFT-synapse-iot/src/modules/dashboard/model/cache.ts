/**
 * dashboard/model/cache.ts
 * Cache theo từng user (Supabase auth user id) — tránh lẫn dữ liệu khi đổi tài khoản cùng trình duyệt.
 */
import type {
  TelemetrySummary, AlertUIModel, TrafficSeriesPoint,
  EdgeNodeRow, AttackDistPoint, ModelStats, HomeInfo,
} from './types';

export interface DashboardCacheState {
  summary:       TelemetrySummary | null;
  recentAlerts:  AlertUIModel[];
  trafficSeries:   TrafficSeriesPoint[];
  homes:           HomeInfo[];
  edgeNodes:       EdgeNodeRow[];
  attackDist:      AttackDistPoint[];
  modelStats:      ModelStats | null;
  networkFlowMetrics?: any[];
  /** Timestamp lần fetch gần nhất (ms). null = chưa fetch lần nào / đã invalidate. */
  fetchedAt:       number | null;
}

const CACHE_TTL_MS = 10 * 1000; // 10 seconds (down from 5 minutes) - ensures near-realtime data

function createEmptyState(): DashboardCacheState {
  return {
    summary:       null,
    recentAlerts:  [],
    trafficSeries: [],
    homes:           [],
    edgeNodes:       [],
    attackDist:      [],
    modelStats:      null,
    networkFlowMetrics: [],
    fetchedAt:       null,
  };
}

/** Một entry cache cho mỗi user đã từng mở dashboard trong phiên tab. */
const cacheByUserId = new Map<string, DashboardCacheState>();

export function getDashboardCacheForUser(userId: string): DashboardCacheState {
  let row = cacheByUserId.get(userId);
  if (!row) {
    row = createEmptyState();
    cacheByUserId.set(userId, row);
  }
  return row;
}

export function isCacheValid(userId: string): boolean {
  const c = cacheByUserId.get(userId);
  if (!c?.fetchedAt) return false;
  return Date.now() - c.fetchedAt < CACHE_TTL_MS;
}

export function writeCache(userId: string, partial: Partial<Omit<DashboardCacheState, 'fetchedAt'>>) {
  const c = getDashboardCacheForUser(userId);
  Object.assign(c, partial);
  c.fetchedAt = Date.now();
}

/** Chỉ xóa TTL (ép refetch); giữ dữ liệu cũ làm fallback tùy nhu cầu. */
export function invalidateCache(userId: string) {
  const c = cacheByUserId.get(userId);
  if (c) c.fetchedAt = null;
}

/** Gọi khi đăng xuất để không giữ bộ nhớ / tránh nhầm sau khi user khác đăng nhập. */
export function clearAllDashboardCaches() {
  cacheByUserId.clear();
}
