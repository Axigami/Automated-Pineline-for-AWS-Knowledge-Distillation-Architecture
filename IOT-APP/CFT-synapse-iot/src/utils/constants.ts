/**
 * Application-wide constants
 */

/** Giới hạn buffer log trên trình duyệt (FR 1.1) */
export const LOG_BUFFER_LIMIT = 1000;

/** Khoảng thời gian polling trạng thái Retrain (FR 4.2), ms */
export const RETRAIN_POLL_INTERVAL_MS = 3000;

/** Khoảng thời gian refresh Fleet status, ms */
export const FLEET_REFRESH_INTERVAL_MS = 30_000;

/** Các nhãn tấn công được nhận diện bởi mô hình */
export const ATTACK_LABELS = [
  'Benign',
  'DoS',
  'PortScan',
  'BruteForce',
  'Infiltration',
] as const;

export type AttackLabel = (typeof ATTACK_LABELS)[number];
