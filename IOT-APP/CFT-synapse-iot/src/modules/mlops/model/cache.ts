/**
 * mlops/model/cache.ts
 * Cache for MLOps data to improve performance
 */
import type { ModelVersionRow, RetrainJobRow } from './types';

export interface MlopsCacheState {
  metrics: ModelVersionRow[];
  activeJob: RetrainJobRow | null;
  fetchedAt: number | null;
}

const CACHE_TTL_MS = 10 * 1000; // 10 seconds

function createEmptyState(): MlopsCacheState {
  return {
    metrics: [],
    activeJob: null,
    fetchedAt: null,
  };
}

const cacheByUserId = new Map<string, MlopsCacheState>();

export function getMlopsCacheForUser(userId: string): MlopsCacheState {
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

export function writeCache(userId: string, partial: Partial<Omit<MlopsCacheState, 'fetchedAt'>>) {
  const c = getMlopsCacheForUser(userId);
  Object.assign(c, partial);
  c.fetchedAt = Date.now();
}

export function invalidateCache(userId: string) {
  const c = cacheByUserId.get(userId);
  if (c) c.fetchedAt = null;
}

export function clearAllMlopsCaches() {
  cacheByUserId.clear();
}
