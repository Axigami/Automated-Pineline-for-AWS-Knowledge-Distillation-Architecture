import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { supabase } from '../../../../core/lib/supabaseClient';
import { useNotifications } from '../../../../core/hooks/useNotifications';
import { useLanguage } from '../../../../core/i18n/LanguageContext';
import { 
  Bell, AlertTriangle, AlertCircle, Info, ChevronRight, 
  Search, Filter, ChevronLeft, ShieldAlert, Wifi, Globe, X
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

interface AlertRow {
  alert_id: string;
  alert_threat_type: string;
  alert_severity: string;
  alert_created_at: string;
  alert_source_ip: string | null;
  alert_target_ip: string | null;
  alert_predicted_label: string | null;
  alert_confidence: number | null;
  alert_verdict_text: string | null;
  alert_node_id: string | null;
}

const getSeverityIcon = (severity: string) => {
  if (severity === 'critical' || severity === 'high') {
    return <AlertTriangle size={16} className="text-rose-400" />;
  }
  if (severity === 'medium') {
    return <AlertCircle size={16} className="text-amber-400" />;
  }
  return <Info size={16} className="text-blue-400" />;
};

const getSeverityBg = (severity: string) => {
  if (severity === 'critical' || severity === 'high') return 'bg-rose-500/10 border-rose-500/20 text-rose-400';
  if (severity === 'medium') return 'bg-amber-500/10 border-amber-500/20 text-amber-400';
  return 'bg-blue-500/10 border-blue-500/20 text-blue-400';
};

export const AlertHistoryPage = () => {
  const { t } = useLanguage();
  const [searchParams, setSearchParams] = useSearchParams();
  const alertIdFromSearch = searchParams.get('alertId');

  const [alerts, setAlerts] = useState<AlertRow[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [page, setPage] = useState(0);
  const limit = 50;
  
  const [selectedAlert, setSelectedAlert] = useState<AlertRow | null>(null);

  const { checkIsRead, markAsRead, markAsUnread } = useNotifications();

  useEffect(() => {
    if (!alertIdFromSearch) return;
    let cancelled = false;
    (async () => {
      const { data, error } = await supabase
        .from('alerts_all')
        .select('*')
        .eq('alert_id', alertIdFromSearch)
        .maybeSingle();

      if (cancelled || error || !data) return;

      const row = data as AlertRow;
      setSelectedAlert(row);
      if (!checkIsRead(row.alert_created_at, row.alert_id)) {
        markAsRead(row.alert_id);
      }

      setSearchParams(
        (prev) => {
          const n = new URLSearchParams(prev);
          n.delete('alertId');
          return n;
        },
        { replace: true },
      );
    })();
    return () => {
      cancelled = true;
    };
    // Chỉ chạy khi có query từ global search — không gắn checkIsRead/markAsRead để tránh chạy lại không cần thiết
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [alertIdFromSearch]);

  const fetchAlerts = async (pageIndex: number) => {
    setIsLoading(true);
    const { data, error } = await supabase
      .from('alerts_all')
      .select('*')
      .order('alert_created_at', { ascending: false })
      .range(pageIndex * limit, (pageIndex + 1) * limit - 1);
      
    if (!error && data) {
      if (pageIndex === 0) {
        setAlerts(data);
      } else {
        setAlerts(prev => [...prev, ...data]);
      }
    }
    setIsLoading(false);
  };

  useEffect(() => {
    fetchAlerts(page);
  }, [page]);

  const handleAlertClick = (alert: AlertRow) => {
    setSelectedAlert(alert);
    if (!checkIsRead(alert.alert_created_at, alert.alert_id)) {
      markAsRead(alert.alert_id);
    }
  };

  const handleToggleReadStatus = (e: React.MouseEvent, alert: AlertRow) => {
    e.stopPropagation();
    const isRead = checkIsRead(alert.alert_created_at, alert.alert_id);
    if (isRead) {
      markAsUnread(alert.alert_id);
    } else {
      markAsRead(alert.alert_id);
    }
  };

  return (
    <div className="max-w-7xl mx-auto flex flex-col h-full relative">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-100 flex items-center gap-3">
            <Bell className="text-blue-500" size={32} />
            {t('alertsHistory', 'title') || 'Alert History'}
          </h1>
          <p className="text-slate-400 mt-2">{t('alertsHistory', 'subTitle') || 'Comprehensive log of all detected security incidents across the edge fleet.'}</p>
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl shadow-xl flex-1 flex flex-col min-h-0 relative">
        {/* Toolbar */}
        <div className="p-4 border-b border-slate-800 flex items-center gap-4 bg-slate-900/50 backdrop-blur-sm z-10 sticky top-0 rounded-t-xl">
          <div className="relative w-64">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
            <input 
              type="text" 
              placeholder={t('alertsHistory', 'searchPlaceholder') || 'Search by IP or type...'}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg pl-10 pr-4 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
            />
          </div>
          <button className="flex items-center gap-2 bg-slate-950 border border-slate-700 text-slate-300 px-4 py-2 rounded-lg text-sm hover:bg-slate-800 transition-colors">
            <Filter size={16} /> {t('alertsHistory', 'filterBtn') || 'Filter'}
          </button>
        </div>

        {/* List */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-0">
          <table className="w-full text-left border-collapse">
            <thead className="bg-slate-950/50 text-slate-400 text-xs uppercase tracking-wider sticky top-0 backdrop-blur z-10">
              <tr>
                <th className="px-6 py-4 font-semibold">{t('alertsHistory', 'colStatus') || 'Status'}</th>
                <th className="px-6 py-4 font-semibold">{t('alertsHistory', 'colTime') || 'Time'}</th>
                <th className="px-6 py-4 font-semibold">{t('alertsHistory', 'colThreatType') || 'Threat Type'}</th>
                <th className="px-6 py-4 font-semibold">{t('alertsHistory', 'colSource') || 'Source'}</th>
                <th className="px-6 py-4 font-semibold">{t('alertsHistory', 'colSeverity') || 'Severity'}</th>
                <th className="px-6 py-4 font-semibold">{t('alertsHistory', 'colAction') || 'Action'}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {alerts.map((alert) => {
                const isRead = checkIsRead(alert.alert_created_at, alert.alert_id);
                return (
                  <tr 
                    key={alert.alert_id} 
                    onClick={() => handleAlertClick(alert)}
                    className={`cursor-pointer transition-colors hover:bg-slate-800/50 ${!isRead ? 'bg-slate-900' : 'opacity-80'}`}
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className={`w-2 h-2 rounded-full ${!isRead ? 'bg-blue-500 animate-pulse' : 'bg-slate-700'}`}></div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-400 font-mono">
                      {new Date(alert.alert_created_at).toLocaleString()}
                    </td>
                    <td className={`px-6 py-4 whitespace-nowrap text-sm font-bold ${!isRead ? 'text-slate-100' : 'text-slate-300'}`}>
                      {alert.alert_threat_type}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-400 font-mono">
                      {alert.alert_source_ip || 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-[10px] font-bold uppercase tracking-widest border ${getSeverityBg(alert.alert_severity)}`}>
                        {getSeverityIcon(alert.alert_severity)}
                        {alert.alert_severity}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <button 
                        onClick={(e) => handleToggleReadStatus(e, alert)}
                        className="text-xs font-bold text-slate-500 hover:text-blue-400 transition-colors uppercase tracking-widest"
                      >
                        {isRead ? (t('alertsHistory', 'markAsUnread') || 'Mark Unread') : (t('alertsHistory', 'markAsRead') || 'Mark Read')}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {isLoading && (
            <div className="p-8 text-center text-slate-500 text-sm">Loading alerts...</div>
          )}

          {!isLoading && alerts.length > 0 && (
            <div className="p-6 text-center">
              <button 
                onClick={() => setPage(p => p + 1)}
                className="bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-semibold py-2 px-6 rounded-lg transition-colors"
              >
                {t('alertsHistory', 'loadOlder') || 'Load Older Alerts'}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Details Side Panel Modal */}
      <AnimatePresence>
        {selectedAlert && (
          <>
            <motion.div 
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="fixed inset-0 bg-slate-950/60 backdrop-blur-sm z-40"
              onClick={() => setSelectedAlert(null)}
            />
            <motion.div 
              initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }} transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed right-0 top-0 bottom-0 w-full max-w-md bg-slate-900 border-l border-slate-800 shadow-2xl z-50 flex flex-col"
            >
              <div className="p-6 border-b border-slate-800 flex items-center justify-between">
                <h2 className="text-xl font-bold flex items-center gap-3 text-slate-100">
                  <ShieldAlert className="text-blue-500" /> {t('alertsHistory', 'alertDetails') || 'Alert Details'}
                </h2>
                <button onClick={() => setSelectedAlert(null)} className="text-slate-500 hover:text-slate-300 p-2"><X size={20} /></button>
              </div>

              <div className="flex-1 overflow-y-auto p-6 space-y-6">
                <div>
                  <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">{t('alertsHistory', 'threatIdent') || 'Threat Identification'}</p>
                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <p className="text-2xl font-black text-rose-400 mb-1">{selectedAlert.alert_threat_type}</p>
                    <p className="text-xs text-slate-400 font-mono">ID: {selectedAlert.alert_id}</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">{t('alertsHistory', 'sourceIp') || 'Source IP'}</p>
                    <div className="flex items-center gap-2 text-slate-200 font-mono text-sm"><Globe size={14} className="text-slate-600"/>{selectedAlert.alert_source_ip || '—'}</div>
                  </div>
                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">{t('alertsHistory', 'targetNode') || 'Target Node'}</p>
                    <div className="flex items-center gap-2 text-slate-200 font-mono text-sm"><Wifi size={14} className="text-slate-600"/>{selectedAlert.alert_target_ip || '—'}</div>
                  </div>
                </div>

                <div>
                  <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">{t('alertsHistory', 'aiVerdict') || 'AI Verdict'}</p>
                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <div className="flex justify-between items-center mb-4 pb-4 border-b border-slate-800">
                      <div>
                        <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">{t('alertsHistory', 'confidence') || 'Confidence'}</p>
                        <p className="text-lg font-black text-emerald-400 font-mono">
                          {selectedAlert.alert_confidence ? `${(selectedAlert.alert_confidence * 100).toFixed(1)}%` : '—'}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">{t('alertsHistory', 'predLabel') || 'Predicted Label'}</p>
                        <p className="text-sm font-bold text-slate-300">{selectedAlert.alert_predicted_label || '—'}</p>
                      </div>
                    </div>
                    <div>
                        <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-1">{t('alertsHistory', 'verifNote') || 'Verification Note'}</p>
                        <p className="text-sm text-slate-400 leading-relaxed">{selectedAlert.alert_verdict_text || 'Automated verification completed without specific notes.'}</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="p-6 border-t border-slate-800 bg-slate-950 text-right">
                <button 
                  onClick={(e) => { handleToggleReadStatus(e, selectedAlert); }}
                  className="bg-slate-800 hover:bg-slate-700 text-slate-300 font-bold py-2 px-6 rounded-lg transition-colors text-sm"
                >
                  {checkIsRead(selectedAlert.alert_created_at, selectedAlert.alert_id) ? (t('alertsHistory', 'markAsUnread') || 'Mark as Unread') : (t('alertsHistory', 'markAsRead') || 'Mark as Read')}
                </button>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
};
