import React, { useState, useEffect } from 'react';
import {
  Sliders,
  AlertCircle,
  Globe,
  Lock,
  Shield,
  Zap,
  Cloud,
  Bell,
  CheckCircle2,
  Save,
  RotateCcw,
  Loader2,
} from 'lucide-react';
import type { HomeSettings, SecurityPrefs } from '../../model/types';
import { useLanguage } from '../../../../core/i18n/LanguageContext';

interface SettingsThresholdsProps {
  home: HomeSettings | null;
  /** Từ users_roles_settings (security_prefs); null → mặc định UI */
  securityPrefs: SecurityPrefs | null;
  isSaving: boolean;
  saveSuccess: boolean;
  error: string | null;
  onSave: (patch: {
    cloudVerificationThreshold: number;
    dataDriftAlertLevel: number;
    twoFactor: boolean;
    smsAlert: boolean;
  }) => void | Promise<void>;
}

/**
 * SettingsThresholds – Presentational Component.
 * Receives DB data via props, displays threshold sliders.
 * Calls onSave() when user clicks Save to write back to Supabase.
 */
const DEFAULT_SECURITY: SecurityPrefs = { twoFactor: true, smsAlert: false };

export const SettingsThresholds: React.FC<SettingsThresholdsProps> = ({
  home,
  securityPrefs,
  isSaving,
  saveSuccess,
  error,
  onSave,
}) => {
  const { t } = useLanguage();
  // --- Local states ---
  const [confidence, setConfidence] = useState(85);
  const [drift, setDrift] = useState(0.15);

  // Security Toggles
  const [is2FAEnabled, setIs2FAEnabled] = useState(true);
  const [isSmsEnabled, setIsSmsEnabled] = useState(false);

  // Baseline states for toggles to detect changes
  const [initial2FA, setInitial2FA] = useState(true);
  const [initialSms, setInitialSms] = useState(false);

  // Sync DB → local khi home hoặc securityPrefs đổi (load/refresh)
  useEffect(() => {
    if (home) {
      setConfidence(home.cloudVerificationThreshold ?? 85);
      setDrift(home.dataDriftAlertLevel ?? 0.15);
    }
    const sec = securityPrefs ?? DEFAULT_SECURITY;
    setIs2FAEnabled(sec.twoFactor);
    setIsSmsEnabled(sec.smsAlert);
    setInitial2FA(sec.twoFactor);
    setInitialSms(sec.smsAlert);
  }, [home, securityPrefs]);

  // Derived state: calculate if there are any pending changes
  const hasChanges =
    confidence !== (home?.cloudVerificationThreshold ?? 85) ||
    drift !== (home?.dataDriftAlertLevel ?? 0.15) ||
    is2FAEnabled !== initial2FA ||
    isSmsEnabled !== initialSms;

  const handleDiscard = () => {
    if (home) {
      setConfidence(home.cloudVerificationThreshold ?? 85);
      setDrift(home.dataDriftAlertLevel ?? 0.15);
      setIs2FAEnabled(initial2FA);
      setIsSmsEnabled(initialSms);
    }
  };

  const handleSave = () => {
    void onSave({
      cloudVerificationThreshold: confidence,
      dataDriftAlertLevel: drift,
      twoFactor: is2FAEnabled,
      smsAlert: isSmsEnabled,
    });
  };

  // Severity label helpers
  const getConfidenceColor = () => {
    if (confidence >= 90) return 'text-emerald-400';
    if (confidence >= 80) return 'text-blue-400';
    if (confidence >= 70) return 'text-amber-400';
    return 'text-red-400';
  };

  const getDriftColor = () => {
    if (drift <= 0.1) return 'text-emerald-400';
    if (drift <= 0.2) return 'text-blue-400';
    if (drift <= 0.35) return 'text-amber-400';
    return 'text-red-400';
  };

  return (
    <div className="space-y-6">
      {/* Removed Redundant Header as requested */}

      {/* Toast: Success */}
      {saveSuccess && (
        <div className="flex items-center gap-2 px-4 py-3 bg-emerald-500/10 border border-emerald-500/30 rounded-lg animate-in fade-in slide-in-from-top-2">
          <CheckCircle2 size={16} className="text-emerald-400" />
          <span className="text-sm text-emerald-300 font-medium">Configuration saved successfully!</span>
        </div>
      )}

      {/* Toast: Error */}
      {error && (
        <div className="flex items-center gap-2 px-4 py-3 bg-red-500/10 border border-red-500/30 rounded-lg">
          <AlertCircle size={16} className="text-red-400" />
          <span className="text-sm text-red-300 font-medium">{error}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ======= PANEL 1: Alert Thresholds ======= */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
          <div className="flex items-center gap-2 mb-8">
            <Sliders size={18} className="text-blue-400" />
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">
              {t('settings', 'thresholds') || 'Alert Thresholds'}
            </h3>
          </div>

          <div className="space-y-10">
            {/* Slider 1: Cloud Verification Trigger */}
            <div className="space-y-4">
              <div className="flex justify-between items-end">
                <div>
                  <h4 className="text-sm font-bold text-slate-200">Cloud Verification Trigger</h4>
                  <p className="text-xs text-slate-500">
                    Threshold for edge model to trigger distillation and remote update
                  </p>
                </div>
                <span className={`text-lg font-mono font-bold ${getConfidenceColor()}`}>
                  {confidence}%
                </span>
              </div>
              <input
                type="range"
                min="50"
                max="99"
                value={confidence}
                onChange={(e) => setConfidence(parseInt(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
              <div className="flex justify-between text-[10px] text-slate-600 font-bold uppercase tracking-widest">
                <span>Low Sensitivity (50%)</span>
                <span>High Sensitivity (99%)</span>
              </div>
            </div>

            {/* Slider 2: Data Drift Alert Level */}
            <div className="space-y-4">
              <div className="flex justify-between items-end">
                <div>
                  <h4 className="text-sm font-bold text-slate-200">Data Drift Alert Level</h4>
                  <p className="text-xs text-slate-500">
                    Alert frequency threshold (number of alerts) to trigger notification
                  </p>
                </div>
                <span className={`text-lg font-mono font-bold ${getDriftColor()}`}>
                  {drift.toFixed(2)}
                </span>
              </div>
              <input
                type="range" /* Maintaining original constraints assuming DB is float */
                min="0.05"
                max="0.5"
                step="0.01"
                value={drift}
                onChange={(e) => setDrift(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
              />
              <div className="flex justify-between text-[10px] text-slate-600 font-bold uppercase tracking-widest">
                <span>Strict (0.05)</span>
                <span>Relaxed (0.50)</span>
              </div>
            </div>
          </div>

          {/* Info banner */}
          <div className="mt-10 p-4 bg-blue-500/5 border border-blue-500/20 rounded-lg flex gap-3">
            <AlertCircle size={18} className="text-blue-400 shrink-0 mt-0.5" />
            <p className="text-xs text-slate-400 leading-relaxed">
              Changing these thresholds will immediately affect deployed <span className="text-blue-400 font-bold">LightGBM Student Models</span> at edge nodes. Edge model distillation will only trigger when the Edge AI confidence is lower than the configured threshold.
            </p>
          </div>
        </div>

        {/* ======= PANEL 2: Connectivity + Security ======= */}
        <div className="space-y-6">
          {/* Infrastructure Connectivity */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
            <div className="flex items-center gap-2 mb-6">
              <Globe size={18} className="text-blue-400" />
              <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">
                Infrastructure Connectivity
              </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl flex items-center gap-4">
                <div className="p-2 bg-emerald-500/10 rounded-lg">
                  <Zap size={20} className="text-emerald-400" />
                </div>
                <div>
                  <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                    Supabase Realtime
                  </p>
                  <div className="flex items-center gap-1.5">
                    <CheckCircle2 size={12} className="text-emerald-400" />
                    <span className="text-xs font-bold text-slate-200">Connected</span>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl flex items-center gap-4">
                <div className="p-2 bg-emerald-500/10 rounded-lg">
                  <Cloud size={20} className="text-emerald-400" />
                </div>
                <div>
                  <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                    Cloud DB (Supabase)
                  </p>
                  <div className="flex items-center gap-1.5">
                    <CheckCircle2 size={12} className="text-emerald-400" />
                    <span className="text-xs font-bold text-slate-200">Synced</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-6 space-y-4">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-500">Supabase Project</span>
                <span className="text-slate-300 font-mono">zpmbvtfptddmbxhmzapz</span>
              </div>
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-500">Home ID</span>
                <span className="text-slate-300 font-mono text-[11px]">
                  {home?.id?.slice(0, 12) ?? '—'}…
                </span>
              </div>
            </div>
          </div>

          {/* Security & Access */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
            <div className="flex items-center gap-2 mb-6">
              <Lock size={18} className="text-blue-400" />
              <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">
                {t('settings', 'securityAccess') || 'Security & Access'}
              </h3>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-slate-950/50 border border-slate-800 rounded-lg">
                <div className="flex items-center gap-3">
                  <Shield size={16} className="text-slate-400" />
                  <span className="text-xs font-bold text-slate-200">2FA Enforcement</span>
                </div>
                <div 
                  className={`w-10 h-5 rounded-full relative cursor-pointer transition-colors ${is2FAEnabled ? 'bg-blue-600' : 'bg-slate-700'}`}
                  onClick={() => setIs2FAEnabled(!is2FAEnabled)}
                >
                  <div className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-all ${is2FAEnabled ? 'right-0.5' : 'left-0.5'}`} />
                </div>
              </div>
              <div className="flex items-center justify-between p-3 bg-slate-950/50 border border-slate-800 rounded-lg">
                <div className="flex items-center gap-3">
                  <Bell size={16} className="text-slate-400" />
                  <span className="text-xs font-bold text-slate-200">Critical Alert SMS</span>
                </div>
                <div 
                  className={`w-10 h-5 rounded-full relative cursor-pointer transition-colors ${isSmsEnabled ? 'bg-blue-600' : 'bg-slate-700'}`}
                  onClick={() => setIsSmsEnabled(!isSmsEnabled)}
                >
                  <div className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-all ${isSmsEnabled ? 'right-0.5' : 'left-0.5'}`} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ======= SAVE / DISCARD BUTTONS ======= */}
      {/* ======= SAVE / REVERT BUTTONS ======= */}
      <div className="flex justify-between items-center mt-6 p-6 bg-slate-900 border border-slate-800 rounded-xl shadow-xl">
        <button
          onClick={() => {
            if(window.confirm('Are you sure you want to reset all parameters to system default safe-presets (85% Confidence, 0.15 Drift)?')) {
               setConfidence(85);
               setDrift(0.15);
               setIs2FAEnabled(true);
               setIsSmsEnabled(false);
            }
          }}
          className="text-slate-500 hover:text-slate-300 border border-slate-800 hover:bg-slate-800 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all flex items-center gap-2"
        >
          <RotateCcw size={16} />
          Reset to Safe Defaults
        </button>

        <div className="flex items-center gap-2 sm:gap-4">
          {hasChanges && (
            <span className="text-xs text-amber-400 animate-pulse font-bold tracking-widest uppercase hidden sm:inline-block">● Unsaved</span>
          )}
          <button
            onClick={handleDiscard}
            disabled={!hasChanges || isSaving}
            className="px-4 sm:px-6 py-2.5 rounded-lg text-sm font-bold text-slate-400 hover:text-slate-200 transition-colors disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-800"
          >
            Discard
          </button>
          <button
            onClick={handleSave}
            disabled={!hasChanges || isSaving}
            className="bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white px-6 sm:px-8 py-2.5 rounded-lg text-sm font-black uppercase tracking-widest transition-all shadow-[0_0_15px_rgba(37,99,235,0.4)] disabled:shadow-none disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isSaving ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                {t('settings', 'saving') || 'Saving...'}
              </>
            ) : (
              <>
                <Save size={16} />
                {t('settings', 'saveSettings') || 'Save Config'}
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
