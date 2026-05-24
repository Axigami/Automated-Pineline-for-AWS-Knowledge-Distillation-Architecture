import React, { useState, useRef, useCallback } from 'react';
import {
  Shield, Activity, RefreshCw, TrendingUp, Server, History,
  FileText, Download, Calendar, Wifi, WifiOff, Loader2,
  CheckCircle2, AlertCircle, Search, X, ChevronDown,
  FileSpreadsheet, FileDown, RotateCcw, Cpu, Thermometer,
  AlertTriangle, Database, Clock, User,
} from 'lucide-react';
import { useFullReportData } from '../../hooks/useFullReportData';
import { useDbStatus } from '../../../../core/useDbStatus';
import { exportToPDF, exportToExcel, exportAuditToCSV } from '../../utils/exportUtils';
import type { ReportFilters } from '../../model/types';
import { useLanguage } from '../../../../core/i18n/LanguageContext';

// ─── Constants ────────────────────────────────────────────────────────────────
const TODAY = new Date().toISOString().slice(0, 10);

const SEVERITY_CLS: Record<string, string> = {
  critical: 'bg-rose-500/15 text-rose-400 border border-rose-500/30',
  high:     'bg-orange-500/15 text-orange-400 border border-orange-500/30',
  medium:   'bg-amber-500/15 text-amber-400 border border-amber-500/30',
  low:      'bg-sky-500/15 text-sky-400 border border-sky-500/30',
};
const ALERT_STATUS_CLS: Record<string, string> = {
  open:          'bg-rose-500/15 text-rose-400',
  investigating: 'bg-amber-500/15 text-amber-400',
  resolved:      'bg-emerald-500/15 text-emerald-400',
  closed:        'bg-slate-500/15 text-slate-400',
};
const NODE_CLS: Record<string, string> = {
  online:  'text-emerald-400',
  offline: 'text-rose-400',
  warning: 'text-amber-400',
};
const JOB_CLS: Record<string, string> = {
  completed: 'text-emerald-400',
  running:   'text-blue-400',
  pending:   'text-amber-400',
  failed:    'text-rose-400',
};
const MODEL_CLS: Record<string, string> = {
  active:   'bg-emerald-500/15 text-emerald-400',
  staging:  'bg-blue-500/15 text-blue-400',
  archived: 'bg-slate-500/15 text-slate-400',
};
const SOURCE_CLS: Record<string, string> = {
  alert:      'text-rose-400',
  retrain:    'text-blue-400',
  deployment: 'text-emerald-400',
};

// ─── Helpers ──────────────────────────────────────────────────────────────────
function fmt(ts?: string | null) {
  if (!ts) return '—';
  return new Date(ts).toLocaleString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}
function pct(v?: number | null) {
  if (v == null) return '—';
  return `${(v * 100).toFixed(1)}%`;
}
function bytes(v?: number | null) {
  if (v == null) return '—';
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)} MB`;
  if (v >= 1_000)     return `${(v / 1_000).toFixed(1)} KB`;
  return `${v} B`;
}
function dur(s?: number | null) {
  if (s == null) return '—';
  if (s >= 60) return `${(s / 60).toFixed(1)} min`;
  return `${s.toFixed(2)} s`;
}

// ─── Sub-components ───────────────────────────────────────────────────────────
function DbBadge() {
  const { connected, loading } = useDbStatus();
  if (loading) return (
    <span className="flex items-center gap-1.5 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
      <Loader2 size={11} className="animate-spin" /> Connecting…
    </span>
  );
  return connected
    ? <span className="flex items-center gap-1.5 text-[10px] font-bold text-emerald-400 uppercase tracking-widest"><Wifi size={11} /> Supabase Connected</span>
    : <span className="flex items-center gap-1.5 text-[10px] font-bold text-rose-400 uppercase tracking-widest"><WifiOff size={11} /> DB Disconnected</span>;
}

function Badge({ cls, label }: { cls: string; label: string }) {
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${cls}`}>
      {label}
    </span>
  );
}

interface StatCardProps {
  icon: React.ElementType; label: string; value: React.ReactNode;
  sub: string; color: 'rose' | 'amber' | 'blue' | 'emerald';
}
function StatCard({ icon: Icon, label, value, sub, color }: StatCardProps) {
  const map = {
    rose:    { bg: 'bg-rose-500/10',    border: 'border-rose-500/20',    icon: 'text-rose-400',    val: 'text-rose-300' },
    amber:   { bg: 'bg-amber-500/10',   border: 'border-amber-500/20',   icon: 'text-amber-400',   val: 'text-amber-300' },
    blue:    { bg: 'bg-blue-500/10',    border: 'border-blue-500/20',    icon: 'text-blue-400',    val: 'text-blue-300' },
    emerald: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/20', icon: 'text-emerald-400', val: 'text-emerald-300' },
  }[color];
  return (
    <div className={`${map.bg} border ${map.border} rounded-xl p-4`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">{label}</span>
        <Icon size={15} className={map.icon} />
      </div>
      <div className={`text-2xl font-black ${map.val}`}>{value}</div>
      <div className="text-[10px] text-slate-500 mt-1">{sub}</div>
    </div>
  );
}

function EmptyTable({ cols, msg = 'No data available for the selected period.' }: { cols: number; msg?: string }) {
  return (
    <tr>
      <td colSpan={cols} className="px-5 py-12 text-center">
        <Database size={28} className="text-slate-700 mx-auto mb-3" />
        <p className="text-slate-500 text-xs">{msg}</p>
      </td>
    </tr>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="px-4 py-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest whitespace-nowrap">{children}</th>;
}

const TABS_CONFIG = [
  { id: 'alerts',  labelKey: 'tabAlerts',   icon: Shield       },
  { id: 'flows',   labelKey: 'tabFlows',    icon: Activity     },
  { id: 'retrain', labelKey: 'tabRetrain',  icon: RefreshCw    },
  { id: 'models',  labelKey: 'tabModels',   icon: TrendingUp   },
  { id: 'fleet',   labelKey: 'tabFleet',    icon: Server       },
  { id: 'audit',   labelKey: 'tabAudit',    icon: History      },
] as const;

// ─── Main Component ───────────────────────────────────────────────────────────
export function ReportingAuditLogs() {
  const { t } = useLanguage();
  const TABS = TABS_CONFIG.map(tab => ({
    ...tab,
    label: t('reports', tab.labelKey) || tab.labelKey,
  }));
  const reportRef   = useRef<HTMLDivElement>(null);
  const [activeTab, setActiveTab]         = useState<string>('alerts');
  const [datePreset, setDatePreset]       = useState('all');
  const [exportOpen, setExportOpen]       = useState(false);
  const [exporting,  setExporting]        = useState(false);
  const [search, setSearch]               = useState('');

  const [filters, setFilters] = useState<ReportFilters>({
    dateRange: { from: '', to: '' },   // empty = no date filter
    homeId: '',
    sections: {
      attackSummary: true, networkFlows: true, modelAccuracy: true,
      retrainJobs:   true, fleetHealth:  true, auditTrail:   true,
    },
  });

  const {
    homes, summaryStats, auditLogs, alerts, networkFlows,
    retrainJobs, modelVersions, fleetHealth, loading, error, refetch,
  } = useFullReportData(filters);

  // ── Preset handler ──────────────────────────────────────────────────────────
  const applyPreset = useCallback((preset: string) => {
    setDatePreset(preset);
    setSearch('');
    if (preset === 'all' || preset === 'custom') {
      if (preset === 'all') setFilters(f => ({ ...f, dateRange: { from: '', to: '' } }));
      return;
    }
    const to = TODAY;
    let from = TODAY;
    if      (preset === '7d')   from = new Date(Date.now() -  7 * 864e5).toISOString().slice(0, 10);
    else if (preset === '30d')  from = new Date(Date.now() - 30 * 864e5).toISOString().slice(0, 10);
    else if (preset === '90d')  from = new Date(Date.now() - 90 * 864e5).toISOString().slice(0, 10);
    else if (preset === 'month') { const d = new Date(); d.setDate(1); from = d.toISOString().slice(0, 10); }
    else if (preset === 'year')  from = `${new Date().getFullYear()}-01-01`;
    setFilters(f => ({ ...f, dateRange: { from, to } }));
  }, []);

  // ── Export handlers ─────────────────────────────────────────────────────────
  const handlePDF = async () => {
    setExporting(true); setExportOpen(false);
    try { await exportToPDF('report-printable', `synapse-report-${TODAY}`); }
    catch (e) { console.error(e); }
    finally   { setExporting(false); }
  };
  const handleExcel = () => {
    setExportOpen(false);
    exportToExcel(
      { filters, summaryStats, alerts, networkFlows, retrainJobs, modelVersions, fleetHealth, auditLogs },
      `synapse-report-${TODAY}`,
    );
  };
  const handleCSV = () => {
    setExportOpen(false);
    exportAuditToCSV(auditLogs, `audit-trail-${TODAY}`);
  };

  // ── Filtered rows ───────────────────────────────────────────────────────────
  const q = search.toLowerCase();
  const filtAlerts  = q ? alerts.filter(r => [r.alert_threat_type, r.alert_severity, r.alert_status, r.alert_source_ip, r.alert_target_ip, r.alert_predicted_label, r.home_name].join(' ').toLowerCase().includes(q)) : alerts;
  const filtFlows   = q ? networkFlows.filter(r => [r.flow_protocol, r.flow_src_ip, r.flow_dst_ip, r.predicted_label, r.home_name].join(' ').toLowerCase().includes(q)) : networkFlows;
  const filtRetrain = q ? retrainJobs.filter(r => [r.job_status, r.audit_user_display_name, r.job_data_range, r.home_name].join(' ').toLowerCase().includes(q)) : retrainJobs;
  const filtModels  = q ? modelVersions.filter(r => [r.version, r.status, r.author, r.model_name].join(' ').toLowerCase().includes(q)) : modelVersions;
  const filtFleet   = q ? fleetHealth.filter(r => [r.node_code, r.status, r.ip_address, r.framework, r.home_name].join(' ').toLowerCase().includes(q)) : fleetHealth;
  const filtAudit   = q ? auditLogs.filter(r => [r.username, r.action, r.resource, r.status, r.source].join(' ').toLowerCase().includes(q)) : auditLogs;

  const tabCounts: Record<string, number> = {
    alerts: filtAlerts.length, flows: filtFlows.length, retrain: filtRetrain.length,
    models: filtModels.length, fleet: filtFleet.length, audit: filtAudit.length,
  };

  const PRESETS = [
    { key: 'all', label: 'All Time' }, { key: '7d', label: '7 Days' },
    { key: '30d', label: '30 Days' },  { key: '90d', label: '90 Days' },
    { key: 'month', label: 'This Month' }, { key: 'year', label: 'This Year' },
    { key: 'custom', label: 'Custom' },
  ];

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="space-y-5">

      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-black text-slate-100 tracking-tight">{t('reports', 'title') || 'Reporting & Audit Logs'}</h2>
          <p className="text-slate-400 text-sm mt-0.5">{t('reports', 'subTitle') || 'Live data from Supabase · Filter by date or home entity · Export PDF / Excel / CSV'}</p>
        </div>
        <div className="flex items-center gap-3">
          <DbBadge />
          <button onClick={refetch} disabled={loading} title="Refresh data"
            className="p-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-slate-200 transition-colors disabled:opacity-40">
            <RotateCcw size={14} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 bg-rose-500/10 border border-rose-500/30 rounded-lg px-4 py-3 text-xs text-rose-400 font-mono">
          <AlertTriangle size={14} /><span>Supabase error: {error}</span>
        </div>
      )}

      {/* Main 2-column grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start">

        {/* ── LEFT Panel (sticky) ── */}
        <aside className="lg:col-span-1 sticky top-6 space-y-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">

            {/* Panel title */}
            <div className="flex items-center gap-2 mb-5">
              <FileText size={15} className="text-blue-400" />
              <h3 className="text-xs font-black text-slate-100 uppercase tracking-widest">Report Generator</h3>
            </div>

            <div className="space-y-5">
              {/* Home select */}
              <div>
                <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1.5">Home / Site</label>
                <select
                  value={filters.homeId}
                  onChange={e => setFilters(f => ({ ...f, homeId: e.target.value }))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-300 focus:outline-none focus:border-blue-500 transition-colors"
                >
                  <option value="">All Sites (Global)</option>
                  {homes.map(h => <option key={h.id} value={h.id}>{h.name} ({h.code})</option>)}
                </select>
              </div>

              {/* Quick presets */}
              <div>
                <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">Date Range</label>
                <div className="grid grid-cols-2 gap-1.5">
                  {PRESETS.map(p => (
                    <button key={p.key} onClick={() => applyPreset(p.key)}
                      className={`py-1.5 px-2 text-[10px] font-bold rounded-lg border transition-all text-center ${
                        datePreset === p.key
                          ? 'bg-blue-600 border-blue-500 text-white'
                          : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-600 hover:text-slate-200'
                      }`}>
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Custom date pickers */}
              {datePreset === 'custom' && (
                <div className="space-y-2">
                  <div>
                    <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">From</label>
                    <div className="flex items-center gap-2 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2">
                      <Calendar size={11} className="text-slate-500 flex-shrink-0" />
                      <input type="date" value={filters.dateRange.from}
                        onChange={e => setFilters(f => ({ ...f, dateRange: { ...f.dateRange, from: e.target.value } }))}
                        className="bg-transparent text-xs text-slate-300 focus:outline-none w-full" />
                    </div>
                  </div>
                  <div>
                    <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">To</label>
                    <div className="flex items-center gap-2 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2">
                      <Calendar size={11} className="text-slate-500 flex-shrink-0" />
                      <input type="date" value={filters.dateRange.to}
                        onChange={e => setFilters(f => ({ ...f, dateRange: { ...f.dateRange, to: e.target.value } }))}
                        className="bg-transparent text-xs text-slate-300 focus:outline-none w-full" />
                    </div>
                  </div>
                </div>
              )}

              {/* Section toggles */}
              <div>
                <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">Export Sections</label>
                <div className="space-y-2">
                  {(Object.keys(filters.sections) as Array<keyof typeof filters.sections>).map(key => {
                    const labels: Record<keyof typeof filters.sections, string> = {
                      attackSummary: 'Security Alerts', networkFlows: 'Network Flows',
                      modelAccuracy: 'Model Versions',  retrainJobs:  'Retrain Jobs',
                      fleetHealth:   'Fleet Health',    auditTrail:   'Audit Trail',
                    };
                    return (
                      <label key={key as string} className="flex items-center gap-2.5 cursor-pointer group">
                        <div className={`w-4 h-4 rounded border flex items-center justify-center transition-colors flex-shrink-0 ${
                          filters.sections[key] ? 'bg-blue-600 border-blue-500' : 'border-slate-700 bg-slate-950'
                        }`} onClick={() => setFilters(f => ({ ...f, sections: { ...f.sections, [key]: !f.sections[key] } }))}>
                          {filters.sections[key] && <CheckCircle2 size={10} className="text-white" />}
                        </div>
                        <span className="text-xs text-slate-400 group-hover:text-slate-200 transition-colors">{labels[key]}</span>
                      </label>
                    );
                  })}
                </div>
              </div>

              {/* Export button */}
              <div className="relative">
                <div className="flex rounded-lg overflow-hidden border border-blue-600 shadow-lg shadow-blue-500/10">
                  <button onClick={handleExcel} disabled={exporting}
                    className="flex-1 flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold py-2.5 px-3 transition-colors disabled:opacity-50">
                    <Download size={13} />
                    {exporting ? 'Exporting…' : 'Download Report'}
                  </button>
                  <button onClick={() => setExportOpen(o => !o)}
                    className="bg-blue-700 hover:bg-blue-600 text-white px-3 border-l border-blue-500 transition-colors">
                    <ChevronDown size={13} className={`transition-transform ${exportOpen ? 'rotate-180' : ''}`} />
                  </button>
                </div>
                {exportOpen && (
                  <div className="absolute bottom-full mb-1 left-0 right-0 bg-slate-800 border border-slate-700 rounded-lg overflow-hidden shadow-2xl z-30">
                    <button onClick={handlePDF} className="w-full flex items-center gap-3 px-4 py-2.5 text-xs text-slate-200 hover:bg-slate-700 transition-colors font-medium">
                      <FileDown size={13} className="text-rose-400" /> Export PDF (visual)
                    </button>
                    <div className="h-px bg-slate-700" />
                    <button onClick={handleExcel} className="w-full flex items-center gap-3 px-4 py-2.5 text-xs text-slate-200 hover:bg-slate-700 transition-colors font-medium">
                      <FileSpreadsheet size={13} className="text-emerald-400" /> Export Excel (.xlsx)
                    </button>
                    <div className="h-px bg-slate-700" />
                    <button onClick={handleCSV} className="w-full flex items-center gap-3 px-4 py-2.5 text-xs text-slate-200 hover:bg-slate-700 transition-colors font-medium">
                      <FileText size={13} className="text-sky-400" /> Export CSV (Audit Trail)
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </aside>

        {/* ── RIGHT Panel ── */}
        <main className="lg:col-span-3 space-y-5">

          {/* Stat Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <StatCard icon={Shield}    color="rose"    label="Total Alerts"
              value={loading ? <Loader2 size={18} className="animate-spin" /> : summaryStats.totalAlerts}
              sub={`${summaryStats.criticalAlerts} critical / high`} />
            <StatCard icon={Activity}  color="amber"   label="Network Flows"
              value={loading ? <Loader2 size={18} className="animate-spin" /> : summaryStats.totalFlows}
              sub={`${summaryStats.anomalyFlows} anomalous`} />
            <StatCard icon={TrendingUp} color="blue"   label="Avg Accuracy"
              value={loading ? <Loader2 size={18} className="animate-spin" /> : `${summaryStats.avgModelAccuracy}%`}
              sub={`${summaryStats.retrainJobs} retrain jobs`} />
            <StatCard icon={Server}    color="emerald" label="Fleet Online"
              value={loading ? <Loader2 size={18} className="animate-spin" /> : `${summaryStats.onlineNodes}/${summaryStats.totalNodes}`}
              sub="edge nodes" />
          </div>

          {/* Tab panel */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl shadow-xl overflow-hidden">

            {/* Search + Tabs row */}
            <div className="border-b border-slate-800">
              <div className="flex overflow-x-auto scrollbar-thin bg-slate-950/40">
                {TABS.map(tab => (
                  <button key={tab.id} onClick={() => { setActiveTab(tab.id); setSearch(''); }}
                    className={`flex items-center gap-1.5 px-4 py-3.5 text-[11px] font-bold whitespace-nowrap border-b-2 transition-all ${
                      activeTab === tab.id
                        ? 'text-blue-400 border-blue-500 bg-blue-500/5'
                        : 'text-slate-500 border-transparent hover:text-slate-300 hover:bg-slate-800/40'
                    }`}>
                    <tab.icon size={12} />
                    {tab.label}
                    <span className={`ml-1 px-1.5 py-0.5 rounded text-[9px] font-black ${
                      activeTab === tab.id ? 'bg-blue-500/20 text-blue-300' : 'bg-slate-800 text-slate-500'
                    }`}>{loading ? '…' : tabCounts[tab.id]}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Search bar */}
            <div className="px-4 py-2.5 border-b border-slate-800 bg-slate-950/20">
              <div className="relative max-w-sm">
                <Search size={12} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                <input value={search} onChange={e => setSearch(e.target.value)}
                  placeholder={`Search ${TABS.find(t => t.id === activeTab)?.label ?? ''}…`}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-8 pr-7 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-blue-500 transition-colors" />
                {search && <button onClick={() => setSearch('')} className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"><X size={11} /></button>}
              </div>
            </div>

            {/* ── Security Alerts ── */}
            {activeTab === 'alerts' && (
              <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
                <table className="w-full text-left">
                  <thead className="sticky top-0 bg-slate-900 z-10 border-b border-slate-800">
                    <tr><Th>Time</Th><Th>Threat Type</Th><Th>Severity</Th><Th>Status</Th><Th>Source IP</Th><Th>Dest IP</Th><Th>Confidence</Th><Th>Label</Th><Th>Home</Th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50">
                    {loading ? <tr><td colSpan={9} className="px-4 py-10 text-center text-xs text-slate-500"><Loader2 size={16} className="animate-spin inline mr-2" />Loading…</td></tr>
                    : filtAlerts.length === 0 ? <EmptyTable cols={9} />
                    : filtAlerts.map(a => (
                      <tr key={a.alert_id} className="hover:bg-slate-800/30 transition-colors">
                        <td className="px-4 py-2.5 text-xs font-mono text-slate-400 whitespace-nowrap">{fmt(a.alert_first_seen_at)}</td>
                        <td className="px-4 py-2.5 text-xs font-bold text-slate-200 whitespace-nowrap">{a.alert_threat_type}</td>
                        <td className="px-4 py-2.5"><Badge cls={SEVERITY_CLS[a.alert_severity] ?? 'text-slate-400'} label={a.alert_severity} /></td>
                        <td className="px-4 py-2.5"><span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded ${ALERT_STATUS_CLS[a.alert_status] ?? 'text-slate-400'}`}>{a.alert_status}</span></td>
                        <td className="px-4 py-2.5 text-xs font-mono text-slate-400">{a.alert_source_ip ?? '—'}</td>
                        <td className="px-4 py-2.5 text-xs font-mono text-slate-400">{a.alert_target_ip ?? '—'}</td>
                        <td className="px-4 py-2.5 text-xs text-slate-300 font-bold">{pct(a.alert_confidence)}</td>
                        <td className="px-4 py-2.5 text-xs text-slate-400 font-mono">{a.alert_predicted_label ?? '—'}</td>
                        <td className="px-4 py-2.5 text-xs text-slate-400">{a.home_name ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* ── Network Flows ── */}
            {activeTab === 'flows' && (
              <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
                <table className="w-full text-left">
                  <thead className="sticky top-0 bg-slate-900 z-10 border-b border-slate-800">
                    <tr><Th>Timestamp</Th><Th>Protocol</Th><Th>Source</Th><Th>Destination</Th><Th>Size</Th><Th>Duration</Th><Th>Anomaly</Th><Th>Label</Th><Th>Confidence</Th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50">
                    {loading ? <tr><td colSpan={9} className="px-4 py-10 text-center text-xs text-slate-500"><Loader2 size={16} className="animate-spin inline mr-2" />Loading…</td></tr>
                    : filtFlows.length === 0 ? <EmptyTable cols={9} />
                    : filtFlows.map(f => (
                      <tr key={f.flow_id} className="hover:bg-slate-800/30 transition-colors">
                        <td className="px-4 py-2.5 text-xs font-mono text-slate-400 whitespace-nowrap">{fmt(f.flow_ts)}</td>
                        <td className="px-4 py-2.5"><span className="text-[10px] font-bold text-slate-300 bg-slate-800 px-2 py-0.5 rounded">{f.flow_protocol ?? '—'}</span></td>
                        <td className="px-4 py-2.5 text-xs font-mono text-slate-400">{f.flow_src_ip ?? '—'}{f.flow_src_port ? `:${f.flow_src_port}` : ''}</td>
                        <td className="px-4 py-2.5 text-xs font-mono text-slate-400">{f.flow_dst_ip ?? '—'}{f.flow_dst_port ? `:${f.flow_dst_port}` : ''}</td>
                        <td className="px-4 py-2.5 text-xs text-slate-300">{bytes(f.flow_total_bytes)}</td>
                        <td className="px-4 py-2.5 text-xs text-slate-400">{dur(f.flow_duration_s)}</td>
                        <td className="px-4 py-2.5">
                          {f.is_anomaly
                            ? <span className="flex items-center gap-1 text-rose-400 text-[10px] font-bold"><AlertCircle size={10} />YES</span>
                            : <span className="text-emerald-500 text-[10px] font-bold">NO</span>}
                        </td>
                        <td className="px-4 py-2.5 text-xs font-mono text-slate-400">{f.predicted_label ?? '—'}</td>
                        <td className="px-4 py-2.5 text-xs font-bold text-slate-300">{pct(f.confidence)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* ── Retrain Jobs ── */}
            {activeTab === 'retrain' && (
              <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
                <table className="w-full text-left">
                  <thead className="sticky top-0 bg-slate-900 z-10 border-b border-slate-800">
                    <tr><Th>Job ID</Th><Th>Status</Th><Th>Progress</Th><Th>Epochs</Th><Th>Created</Th><Th>Finished</Th><Th>KD</Th><Th>Performed By</Th><Th>Home</Th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50">
                    {loading ? <tr><td colSpan={9} className="px-4 py-10 text-center text-xs text-slate-500"><Loader2 size={16} className="animate-spin inline mr-2" />Loading…</td></tr>
                    : filtRetrain.length === 0 ? <EmptyTable cols={9} />
                    : filtRetrain.map(j => (
                      <tr key={j.job_id} className="hover:bg-slate-800/30 transition-colors">
                        <td className="px-4 py-2.5 text-xs font-mono text-slate-500">{j.job_id.slice(0, 8)}…</td>
                        <td className="px-4 py-2.5"><span className={`text-[10px] font-black uppercase ${JOB_CLS[j.job_status] ?? 'text-slate-400'}`}>{j.job_status}</span></td>
                        <td className="px-4 py-2.5">
                          <div className="flex items-center gap-2">
                            <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                              <div className="h-full bg-blue-500 rounded-full" style={{ width: `${j.job_progress_percent ?? 0}%` }} />
                            </div>
                            <span className="text-xs text-slate-400">{j.job_progress_percent ?? 0}%</span>
                          </div>
                        </td>
                        <td className="px-4 py-2.5 text-xs text-slate-300">{j.job_epochs ?? '—'}</td>
                        <td className="px-4 py-2.5 text-xs font-mono text-slate-400 whitespace-nowrap">{fmt(j.job_created_at)}</td>
                        <td className="px-4 py-2.5 text-xs font-mono text-slate-400 whitespace-nowrap">{fmt(j.job_finished_at)}</td>
                        <td className="px-4 py-2.5 text-xs text-slate-400">{j.job_knowledge_distillation ? <span className="text-purple-400 font-bold">Yes</span> : 'No'}</td>
                        <td className="px-4 py-2.5 text-xs text-slate-300">{j.audit_user_display_name ?? '—'}</td>
                        <td className="px-4 py-2.5 text-xs text-slate-400">{j.home_name ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* ── Model Versions ── */}
            {activeTab === 'models' && (
              <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
                <table className="w-full text-left">
                  <thead className="sticky top-0 bg-slate-900 z-10 border-b border-slate-800">
                    <tr><Th>Model</Th><Th>Version</Th><Th>Status</Th><Th>Accuracy</Th><Th>F1</Th><Th>Precision</Th><Th>Recall</Th><Th>FPR</Th><Th>Latency</Th><Th>Author</Th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50">
                    {loading ? <tr><td colSpan={10} className="px-4 py-10 text-center text-xs text-slate-500"><Loader2 size={16} className="animate-spin inline mr-2" />Loading…</td></tr>
                    : filtModels.length === 0 ? <EmptyTable cols={10} />
                    : filtModels.map(m => (
                      <tr key={m.id} className="hover:bg-slate-800/30 transition-colors">
                        <td className="px-4 py-2.5 text-xs font-bold text-slate-200">{m.model_name ?? '—'}</td>
                        <td className="px-4 py-2.5 text-xs font-mono text-blue-400">{m.version}</td>
                        <td className="px-4 py-2.5"><Badge cls={MODEL_CLS[m.status] ?? 'text-slate-400'} label={m.status} /></td>
                        <td className="px-4 py-2.5 text-xs font-black text-emerald-400">{pct(m.accuracy)}</td>
                        <td className="px-4 py-2.5 text-xs text-slate-300">{m.f1_score != null ? m.f1_score.toFixed(3) : '—'}</td>
                        <td className="px-4 py-2.5 text-xs text-slate-300">{m.precision != null ? m.precision.toFixed(3) : '—'}</td>
                        <td className="px-4 py-2.5 text-xs text-slate-300">{m.recall != null ? m.recall.toFixed(3) : '—'}</td>
                        <td className="px-4 py-2.5 text-xs text-slate-400">{m.false_positive_rate != null ? m.false_positive_rate.toFixed(3) : '—'}</td>
                        <td className="px-4 py-2.5 text-xs text-slate-400">{m.latency_ms != null ? `${m.latency_ms}ms` : '—'}</td>
                        <td className="px-4 py-2.5 text-xs text-slate-400">{m.author ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* ── Fleet Health ── */}
            {activeTab === 'fleet' && (
              <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
                <table className="w-full text-left">
                  <thead className="sticky top-0 bg-slate-900 z-10 border-b border-slate-800">
                    <tr><Th>Node</Th><Th>Status</Th><Th>IP Address</Th><Th>CPU</Th><Th>RAM</Th><Th>Temp °C</Th><Th>Latency</Th><Th>Framework</Th><Th>Model</Th><Th>Last Seen</Th><Th>Home</Th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50">
                    {loading ? <tr><td colSpan={11} className="px-4 py-10 text-center text-xs text-slate-500"><Loader2 size={16} className="animate-spin inline mr-2" />Loading…</td></tr>
                    : filtFleet.length === 0 ? <EmptyTable cols={11} />
                    : filtFleet.map(n => (
                      <tr key={n.id} className="hover:bg-slate-800/30 transition-colors">
                        <td className="px-4 py-2.5 text-xs font-bold text-slate-200 font-mono">{n.node_code}</td>
                        <td className="px-4 py-2.5">
                          <span className={`flex items-center gap-1.5 text-[10px] font-black uppercase ${NODE_CLS[n.status] ?? 'text-slate-400'}`}>
                            <span className={`w-1.5 h-1.5 rounded-full ${n.status === 'online' ? 'bg-emerald-400' : n.status === 'warning' ? 'bg-amber-400' : 'bg-rose-400'}`} />
                            {n.status}
                          </span>
                        </td>
                        <td className="px-4 py-2.5 text-xs font-mono text-slate-400">{n.ip_address ?? '—'}</td>
                        <td className="px-4 py-2.5">
                          {n.current_cpu_percent != null ? (
                            <div className="flex items-center gap-1.5">
                              <div className="w-10 h-1 bg-slate-800 rounded-full overflow-hidden">
                                <div className={`h-full rounded-full ${n.current_cpu_percent > 80 ? 'bg-rose-500' : n.current_cpu_percent > 60 ? 'bg-amber-500' : 'bg-emerald-500'}`} style={{ width: `${n.current_cpu_percent}%` }} />
                              </div>
                              <span className="text-xs text-slate-300">{n.current_cpu_percent.toFixed(0)}%</span>
                            </div>
                          ) : <span className="text-slate-600">—</span>}
                        </td>
                        <td className="px-4 py-2.5">
                          {n.current_ram_percent != null ? (
                            <div className="flex items-center gap-1.5">
                              <div className="w-10 h-1 bg-slate-800 rounded-full overflow-hidden">
                                <div className={`h-full rounded-full ${n.current_ram_percent > 85 ? 'bg-rose-500' : n.current_ram_percent > 65 ? 'bg-amber-500' : 'bg-blue-500'}`} style={{ width: `${n.current_ram_percent}%` }} />
                              </div>
                              <span className="text-xs text-slate-300">{n.current_ram_percent.toFixed(0)}%</span>
                            </div>
                          ) : <span className="text-slate-600">—</span>}
                        </td>
                        <td className="px-4 py-2.5 text-xs text-slate-300">{n.current_temperature_c != null ? `${n.current_temperature_c.toFixed(1)}°` : '—'}</td>
                        <td className="px-4 py-2.5 text-xs text-slate-400">{n.current_latency_ms != null ? `${n.current_latency_ms}ms` : '—'}</td>
                        <td className="px-4 py-2.5 text-xs text-slate-400">{n.framework ?? '—'}</td>
                        <td className="px-4 py-2.5 text-xs font-mono text-slate-500">{n.model_version_text ?? '—'}</td>
                        <td className="px-4 py-2.5 text-xs font-mono text-slate-400 whitespace-nowrap">{fmt(n.last_seen_at)}</td>
                        <td className="px-4 py-2.5 text-xs text-slate-400">{n.home_name ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* ── Audit Trail ── */}
            {activeTab === 'audit' && (
              <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
                <table className="w-full text-left">
                  <thead className="sticky top-0 bg-slate-900 z-10 border-b border-slate-800">
                    <tr><Th>Timestamp</Th><Th>User</Th><Th>Action</Th><Th>Resource</Th><Th>Status</Th><Th>Source</Th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50">
                    {loading ? <tr><td colSpan={6} className="px-4 py-10 text-center text-xs text-slate-500"><Loader2 size={16} className="animate-spin inline mr-2" />Loading…</td></tr>
                    : filtAudit.length === 0 ? <EmptyTable cols={6} />
                    : filtAudit.map((l, i) => (
                      <tr key={i} className="hover:bg-slate-800/30 transition-colors">
                        <td className="px-4 py-2.5 text-xs font-mono text-slate-400 whitespace-nowrap">{l.timestamp}</td>
                        <td className="px-4 py-2.5">
                          <div className="flex items-center gap-1.5">
                            <div className="w-5 h-5 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center flex-shrink-0">
                              <User size={9} className="text-slate-500" />
                            </div>
                            <span className="text-xs font-bold text-slate-200">{l.username}</span>
                          </div>
                        </td>
                        <td className="px-4 py-2.5 text-xs font-mono font-bold text-slate-300">{l.action}</td>
                        <td className="px-4 py-2.5 text-xs font-mono text-slate-400 max-w-[200px] truncate">{l.resource}</td>
                        <td className="px-4 py-2.5">
                          {l.status?.toLowerCase() === 'success' || l.status?.toLowerCase() === 'completed'
                            ? <span className="flex items-center gap-1 text-emerald-400 text-[10px] font-bold"><CheckCircle2 size={10} />{l.status}</span>
                            : <span className="text-[10px] font-bold text-slate-400">{l.status}</span>}
                        </td>
                        <td className="px-4 py-2.5">
                          <span className={`text-[10px] font-black uppercase ${SOURCE_CLS[l.source] ?? 'text-slate-400'}`}>{l.source}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

          </div>{/* end tab panel */}
        </main>
      </div>

      {/* Hidden printable area for PDF */}
      <div id="report-printable" className="fixed -top-[9999px] left-0 w-[900px] p-8 space-y-6" ref={reportRef} style={{ backgroundColor: '#020617', color: '#f1f5f9' }}>
        <div className="pb-4" style={{ borderBottom: '1px solid #334155' }}>
          <h1 className="text-xl font-black" style={{ color: '#f1f5f9' }}>Synapse IoT — Security Report</h1>
          <p className="text-xs mt-1" style={{ color: '#94a3b8' }}>
            Period: {filters.dateRange.from && filters.dateRange.to ? `${filters.dateRange.from} → ${filters.dateRange.to}` : 'All Time'} ·
            Generated: {new Date().toLocaleString('en-US')}
          </p>
        </div>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: 'Total Alerts',  value: summaryStats.totalAlerts },
            { label: 'Network Flows', value: summaryStats.totalFlows },
            { label: 'Avg Accuracy',  value: `${summaryStats.avgModelAccuracy}%` },
            { label: 'Fleet Online',  value: `${summaryStats.onlineNodes}/${summaryStats.totalNodes}` },
          ].map(c => (
            <div key={c.label} className="rounded-lg p-3 border" style={{ backgroundColor: '#0f172a', borderColor: '#1e293b' }}>
              <div className="text-[10px] uppercase" style={{ color: '#64748b' }}>{c.label}</div>
              <div className="text-xl font-black" style={{ color: '#f1f5f9' }}>{c.value}</div>
            </div>
          ))}
        </div>
        {filters.sections.attackSummary && alerts.length > 0 && (
          <div>
            <h2 className="text-sm font-black mb-2" style={{ color: '#e2e8f0' }}>Security Alerts ({alerts.length})</h2>
            <table className="w-full text-xs border-collapse">
              <thead><tr style={{ backgroundColor: '#1e293b' }}>{['Time','Threat','Severity','Status','Src IP','Dst IP','Confidence'].map(h => <th key={h} className="px-2 py-1 text-left text-[10px] font-bold" style={{ color: '#94a3b8' }}>{h}</th>)}</tr></thead>
              <tbody>{alerts.slice(0, 20).map(a => <tr key={a.alert_id} style={{ borderTop: '1px solid #1e293b' }}><td className="px-2 py-1 font-mono" style={{ color: '#94a3b8' }}>{fmt(a.alert_first_seen_at)}</td><td className="px-2 py-1" style={{ color: '#e2e8f0' }}>{a.alert_threat_type}</td><td className="px-2 py-1" style={{ color: '#cbd5e1' }}>{a.alert_severity}</td><td className="px-2 py-1" style={{ color: '#cbd5e1' }}>{a.alert_status}</td><td className="px-2 py-1 font-mono" style={{ color: '#94a3b8' }}>{a.alert_source_ip ?? '—'}</td><td className="px-2 py-1 font-mono" style={{ color: '#94a3b8' }}>{a.alert_target_ip ?? '—'}</td><td className="px-2 py-1" style={{ color: '#cbd5e1' }}>{pct(a.alert_confidence)}</td></tr>)}</tbody>
            </table>
          </div>
        )}
        {filters.sections.auditTrail && auditLogs.length > 0 && (
          <div>
            <h2 className="text-sm font-black mb-2" style={{ color: '#e2e8f0' }}>Audit Trail ({auditLogs.length})</h2>
            <table className="w-full text-xs border-collapse">
              <thead><tr style={{ backgroundColor: '#1e293b' }}>{['Timestamp','User','Action','Resource','Status','Source'].map(h => <th key={h} className="px-2 py-1 text-left text-[10px] font-bold" style={{ color: '#94a3b8' }}>{h}</th>)}</tr></thead>
              <tbody>{auditLogs.slice(0, 20).map((l, i) => <tr key={i} style={{ borderTop: '1px solid #1e293b' }}><td className="px-2 py-1 font-mono" style={{ color: '#94a3b8' }}>{l.timestamp}</td><td className="px-2 py-1" style={{ color: '#e2e8f0' }}>{l.username}</td><td className="px-2 py-1" style={{ color: '#cbd5e1' }}>{l.action}</td><td className="px-2 py-1 font-mono max-w-[160px] truncate" style={{ color: '#94a3b8' }}>{l.resource}</td><td className="px-2 py-1" style={{ color: '#cbd5e1' }}>{l.status}</td><td className="px-2 py-1" style={{ color: '#94a3b8' }}>{l.source}</td></tr>)}</tbody>
            </table>
          </div>
        )}
      </div>

    </div>
  );
}

// Re-export as ReportsPage for routing compatibility
export { ReportingAuditLogs as ReportsPage };
