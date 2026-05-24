import { useEffect, useCallback, useState } from 'react';
import { supabase } from '../../../core/lib/supabaseClient';
import type { HomeSettings, UserRoleSetting, SecurityPrefs } from '../model/types';

export const SETTINGS_SECURITY_KEY = 'security_prefs';

function parseSecurityPrefs(row: {
  setting_key: string | null;
  setting_value_text: string | null;
}): SecurityPrefs | null {
  if (row.setting_key !== SETTINGS_SECURITY_KEY || !row.setting_value_text) return null;
  try {
    const o = JSON.parse(row.setting_value_text) as Partial<SecurityPrefs>;
    if (typeof o.twoFactor !== 'boolean' || typeof o.smsAlert !== 'boolean') return null;
    return { twoFactor: o.twoFactor, smsAlert: o.smsAlert };
  } catch {
    return null;
  }
}

/**
 * useSettings – queries homes (thresholds) + users_roles_settings.
 * Update threshold bằng cách update bảng homes.
 */
export function useSettings() {
  const [homes, setHomes] = useState<HomeSettings[]>([]);
  const [userSettings, setUserSettings] = useState<UserRoleSetting[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState(false);

  const fetchData = useCallback(async () => {
    setIsLoading(true);

    const [homesRes, usersRes] = await Promise.all([
      supabase
        .from('homes' as any)
        .select('id, code, name, region, cloud_verification_confidence_threshold, data_drift_alert_level'),
      supabase
        .from('users_roles_settings' as any)
        .select('user_id, user_email, user_display_name, role_code, role_name, setting_key, setting_value_number, setting_value_text, setting_updated_at'),
    ]);

    if (homesRes.data) {
      const parsedHomes: HomeSettings[] = (homesRes.data as any[]).map((r: any) => ({
        id: r.id,
        code: r.code,
        name: r.name,
        region: r.region,
        cloudVerificationThreshold: r.cloud_verification_confidence_threshold,
        dataDriftAlertLevel: r.data_drift_alert_level,
      }));
      setHomes(parsedHomes);
    }

    if (usersRes.data) {
      setUserSettings(
        (usersRes.data as any[]).map((r: any) => {
          const base = {
            userId: r.user_id,
            email: r.user_email,
            displayName: r.user_display_name,
            roleCode: r.role_code,
            roleName: r.role_name,
            settingKey: r.setting_key,
            settingValueNumber: r.setting_value_number,
            settingValueText: r.setting_value_text,
            settingUpdatedAt: r.setting_updated_at,
          };
          const securityPrefs = parseSecurityPrefs({
            setting_key: r.setting_key,
            setting_value_text: r.setting_value_text,
          });
          return { ...base, securityPrefs: securityPrefs ?? undefined };
        })
      );
    }

    if (homesRes.error || usersRes.error) {
      setError('Không thể tải cấu hình hệ thống');
    }

    setIsLoading(false);
  }, []);

  const updateHomeThreshold = useCallback(
    async (homeId: string, patch: Partial<Pick<HomeSettings, 'cloudVerificationThreshold' | 'dataDriftAlertLevel'>>) => {
      setIsSaving(true);
      const payload: Record<string, number> = {};
      if (patch.cloudVerificationThreshold !== undefined) {
        payload.cloud_verification_confidence_threshold = patch.cloudVerificationThreshold;
      }
      if (patch.dataDriftAlertLevel !== undefined) {
        payload.data_drift_alert_level = patch.dataDriftAlertLevel;
      }
      if (Object.keys(payload).length === 0) {
        setIsSaving(false);
        return;
      }

      const db = supabase as any;
      const res = await db.from('homes').update(payload).eq('id', homeId);
      const updateError = res.error;

      if (updateError) {
        setError('Không thể lưu cấu hình: ' + updateError.message);
      } else {
        setSaveSuccess(true);
        setTimeout(() => setSaveSuccess(false), 3000);
        fetchData();
      }
      setIsSaving(false);
    },
    [fetchData]
  );

  /** Cột users_roles_settings (một dòng / user_id): security_prefs trong setting_value_text */
  const updateUserSecurityPrefs = useCallback(
    async (userId: string, prefs: SecurityPrefs) => {
      setIsSaving(true);
      setError(null);
      const now = new Date().toISOString();
      const text = JSON.stringify(prefs);

      const { data: existing, error: selErr } = await supabase
        .from('users_roles_settings' as any)
        .select('user_id')
        .eq('user_id', userId)
        .maybeSingle();

      if (selErr) {
        setError('Không thể đọc cấu hình người dùng: ' + selErr.message);
        setIsSaving(false);
        return;
      }

      let updateError = null as { message: string } | null;
      if (existing) {
        const res = await supabase
          .from('users_roles_settings' as any)
          .update({
            setting_key: SETTINGS_SECURITY_KEY,
            setting_value_text: text,
            setting_updated_at: now,
          })
          .eq('user_id', userId);
        updateError = res.error;
      } else {
        const res = await supabase.from('users_roles_settings' as any).insert({
          user_id: userId,
          setting_key: SETTINGS_SECURITY_KEY,
          setting_value_text: text,
          setting_updated_at: now,
        });
        updateError = res.error;
      }

      if (updateError) {
        setError('Không thể lưu tùy chọn bảo mật: ' + updateError.message);
      } else {
        setSaveSuccess(true);
        setTimeout(() => setSaveSuccess(false), 3000);
        fetchData();
      }
      setIsSaving(false);
    },
    [fetchData]
  );

  /** Một lần lưu: homes + users_roles_settings (tránh isSaving chồng chéo) */
  const saveFromSettingsPage = useCallback(
    async (opts: {
      homeId: string | null;
      thresholds: { cloudVerificationThreshold: number; dataDriftAlertLevel: number } | null;
      userId: string | null;
      security: SecurityPrefs;
    }) => {
      setIsSaving(true);
      setError(null);
      try {
        if (opts.homeId && opts.thresholds) {
          const { error: e1 } = await supabase
            .from('homes' as any)
            .update({
              cloud_verification_confidence_threshold: opts.thresholds.cloudVerificationThreshold,
              data_drift_alert_level: opts.thresholds.dataDriftAlertLevel,
            })
            .eq('id', opts.homeId);
          if (e1) throw new Error('Ngưỡng Home: ' + e1.message);
        }

        if (opts.userId) {
          const now = new Date().toISOString();
          const text = JSON.stringify(opts.security);
          const { data: existing, error: selErr } = await supabase
            .from('users_roles_settings' as any)
            .select('user_id')
            .eq('user_id', opts.userId)
            .maybeSingle();
          if (selErr) throw new Error('Đọc user settings: ' + selErr.message);

          if (existing) {
            const { error: e2 } = await supabase
              .from('users_roles_settings' as any)
              .update({
                setting_key: SETTINGS_SECURITY_KEY,
                setting_value_text: text,
                setting_updated_at: now,
              })
              .eq('user_id', opts.userId);
            if (e2) throw new Error('Bảo mật: ' + e2.message);
          } else {
            const { error: e2 } = await supabase.from('users_roles_settings' as any).insert({
              user_id: opts.userId,
              setting_key: SETTINGS_SECURITY_KEY,
              setting_value_text: text,
              setting_updated_at: now,
            });
            if (e2) throw new Error('Bảo mật: ' + e2.message);
          }
        }

        setSaveSuccess(true);
        setTimeout(() => setSaveSuccess(false), 3000);
        await fetchData();
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : 'Lưu thất bại');
      } finally {
        setIsSaving(false);
      }
    },
    [fetchData]
  );

  useEffect(() => { fetchData(); }, [fetchData]);

  return {
    homes,
    setHomes,
    userSettings,
    setUserSettings,
    isLoading,
    setIsLoading,
    isSaving,
    setIsSaving,
    error,
    setError,
    saveSuccess,
    setSaveSuccess,
    updateHomeThreshold,
    updateUserSecurityPrefs,
    saveFromSettingsPage,
    refresh: fetchData,
  };
}
