/**
 * formatDate – Định dạng timestamp sang chuỗi ngày giờ Việt Nam.
 */
export function formatDate(isoString: string, locale = 'vi-VN'): string {
  return new Date(isoString).toLocaleString(locale);
}

/**
 * formatRelativeTime – "X giây / phút / giờ / ngày trước"
 */
export function formatRelativeTime(isoString: string): string {
  const seconds = Math.floor((Date.now() - new Date(isoString).getTime()) / 1000);
  if (seconds < 60) return `${seconds} giây trước`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} phút trước`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} giờ trước`;
  return `${Math.floor(seconds / 86400)} ngày trước`;
}

/**
 * formatPct – Số thập phân 0–1 thành chuỗi phần trăm "xx%"
 */
export function formatPct(value: number, decimals = 0): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * clamp – Giới hạn giá trị trong đoạn [min, max]
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
