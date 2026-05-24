/**
 * Core Auth DTOs & Global State Types
 * Dùng chung cho toàn bộ ứng dụng.
 */

// ---- User & Role ----
export type UserRole = 'SOC_ANALYST_L1' | 'THREAT_HUNTER_L2' | 'ML_ENGINEER_L3';

export interface User {
  id: string;
  username: string;
  email: string;
  role: UserRole;
  avatarUrl?: string;
}

// ---- Auth DTOs ----
export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  accessToken: string;
  refreshToken: string;
  user: User;
}

// ---- Global App State ----
export interface AppState {
  currentUser: User | null;
  isAuthenticated: boolean;
  selectedLocation: string;
}
