/**
 * Settings – Types aligned to homes + users_roles_settings
 */

export interface HomeSettings {
  id: string;
  code: string;
  name: string;
  region: string | null;
  cloudVerificationThreshold: number | null;  // cloud_verification_confidence_threshold
  dataDriftAlertLevel: number | null;          // data_drift_alert_level
}

/** Lưu trong users_roles_settings.setting_value_text khi setting_key = security_prefs */
export interface SecurityPrefs {
  twoFactor: boolean;
  smsAlert: boolean;
}

export interface UserRoleSetting {
  userId: string;
  email: string | null;
  displayName: string | null;
  roleCode: string | null;
  roleName: string | null;
  settingKey: string | null;
  settingValueNumber: number | null;
  settingValueText: string | null;
  settingUpdatedAt: string | null;
  /** Parse từ setting_value_text nếu setting_key là security_prefs */
  securityPrefs?: SecurityPrefs | null;
}
