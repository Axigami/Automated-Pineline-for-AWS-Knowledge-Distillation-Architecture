import React, { useState, useEffect, useMemo } from 'react';
import { Settings } from 'lucide-react';
import { useAuthContext } from '../../../../core/auth/AuthProvider';
import { useSettings } from '../../controller';
import { SettingsThresholds } from '../components/SettingsThresholds';

const DEFAULT_HOME = {
  id: "11111111-1111-1111-1111-111111111111",
  code: "HOME001",
  name: "System Default Home",
  region: "Edge Network",
  cloudVerificationThreshold: 85,
  dataDriftAlertLevel: 0.15
};

/**
 * SettingsPage – Container Component.
 * Gọi useSettings() để lấy data + actions từ Supabase,
 * sau đó truyền xuống các Presentational Components qua props.
 */
const SettingsPage: React.FC = () => {
  const { user } = useAuthContext();
  const {
    homes,
    userSettings,
    isLoading,
    isSaving,
    error,
    saveSuccess,
    saveFromSettingsPage,
    refresh,
  } = useSettings();

  const [selectedHomeId, setSelectedHomeId] = useState<string | null>(null);

  const securityPrefsForUser = useMemo(() => {
    if (!user) return null;
    const row = userSettings.find((u) => u.userId === user.id);
    return row?.securityPrefs ?? null;
  }, [user, userSettings]);

  const resolvedHome = useMemo(() => {
    if (homes.length === 0) return null;
    return homes.find((h) => h.id === selectedHomeId) ?? homes[0];
  }, [homes, selectedHomeId]);

  useEffect(() => {
    if (homes.length > 0 && !selectedHomeId) {
      setSelectedHomeId(homes[0].id);
    }
  }, [homes, selectedHomeId]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex flex-col items-center gap-4">
          <div className="w-10 h-10 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-slate-400 font-medium">Loading system configuration...</p>
        </div>
      </div>
    );
  }

  if (error && homes.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center space-y-3">
          <p className="text-red-400 text-sm font-semibold">{error}</p>
          <button
            onClick={refresh}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm rounded-lg transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const activeHome = resolvedHome ?? DEFAULT_HOME;

  return (
    <div className="space-y-8">
      {/* Debug view to see what's happening */}
      {/* <pre className="text-xs text-white bg-black p-2">
        {JSON.stringify({ homesLength: homes.length, activeHome }, null, 2)}
      </pre> */}
      
      <div className="flex flex-col md:flex-row md:items-start justify-between gap-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-100 flex items-center gap-3">
            <Settings className="text-blue-500" size={32} />
            System & Edge Settings
          </h1>
          <p className="text-slate-400 mt-2">Manage edge fleet parameters, Cloud ML verification defaults, and automated defense thresholds.</p>
        </div>

        {homes.length > 0 && (
          <div className="flex items-center gap-3 bg-slate-900 border border-slate-800 px-4 py-2.5 rounded-xl shadow-lg shrink-0">
            <span className="text-xs font-bold text-slate-500 uppercase tracking-widest">Target Node:</span>
            <select 
              value={selectedHomeId || ''} 
              onChange={(e) => setSelectedHomeId(e.target.value)}
              className="bg-slate-950 border border-slate-700 text-slate-200 text-sm font-bold rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue-500 hover:border-slate-600 transition-colors cursor-pointer"
            >
              {homes.map(h => (
                <option key={h.id} value={h.id}>{h.name} ({h.code})</option>
              ))}
            </select>
          </div>
        )}
      </div>
      
      {/* Threshold Configuration – kết nối DB thật */}
      <SettingsThresholds
        home={activeHome}
        securityPrefs={securityPrefsForUser}
        isSaving={isSaving}
        saveSuccess={saveSuccess}
        error={error}
        onSave={(patch) =>
          saveFromSettingsPage({
            homeId: resolvedHome?.id ?? null,
            thresholds: resolvedHome
              ? {
                  cloudVerificationThreshold: patch.cloudVerificationThreshold,
                  dataDriftAlertLevel: patch.dataDriftAlertLevel,
                }
              : null,
            userId: user?.id ?? null,
            security: { twoFactor: patch.twoFactor, smsAlert: patch.smsAlert },
          })
        }
      />
    </div>
  );
};

export default SettingsPage;
