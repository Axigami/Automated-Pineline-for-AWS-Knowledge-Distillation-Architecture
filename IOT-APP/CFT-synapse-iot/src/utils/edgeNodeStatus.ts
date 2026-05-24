/**
 * Chuẩn hóa trạng thái edge node cho UI: chỉ **online** (xanh) vs **offline** (đỏ).
 * Mọi giá trị DB khác (warning, running, degraded, …) được coi là **online** nếu không thuộc nhóm offline.
 */
export type EdgePresence = 'online' | 'offline';

const OFFLINE_TOKENS = ['offline', 'down', 'disconnected', 'inactive', 'stopped', 'lost', 'unreachable'] as const;

export function normalizeEdgePresence(statusRaw: string | null | undefined): EdgePresence {
  const s = (statusRaw ?? '').toLowerCase().trim();
  if (!s) return 'offline';
  if (OFFLINE_TOKENS.some((t) => s === t || s.includes(t))) return 'offline';
  return 'online';
}
