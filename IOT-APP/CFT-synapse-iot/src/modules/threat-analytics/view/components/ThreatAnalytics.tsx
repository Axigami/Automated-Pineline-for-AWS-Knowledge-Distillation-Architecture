import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Search,
  Clock,
  Filter,
  Download,
  CheckCircle2,
  XCircle,
  ExternalLink,
  BarChart3,
  PieChart as PieChartIcon,
  RefreshCw,
  AlertCircle,
  X,
} from 'lucide-react';
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { motion, AnimatePresence } from 'motion/react';
import { useLanguage } from '../../../../core/i18n/LanguageContext';
import type {
  FlowUIModel,
  LabelAggregation,
  TopAttacker,
  TimelinePoint,
  FlowQueryParams,
  LabelFeedbackRequest,
} from '../../model/types';

// ─── Bảng màu nhãn (đồng bộ với adapter) ─────────────────────────────────────
const LABEL_COLOR: Record<string, string> = {
  PortScan: '#f43f5e',
  DDoS: '#3b82f6',
  Botnet: '#fbbf24',
  DoS: '#a855f7',
  BruteForce: '#f97316',
  'Web attack': '#06b6d4',
  'Web Attack': '#06b6d4',
  Benign: '#22c55e',
};
const DEFAULT_LABEL_COLOR = '#64748b';

function labelColor(label: string): { bg: string; text: string } {
  const map: Record<string, string> = {
    PortScan: 'bg-rose-500/10 text-rose-400',
    DDoS: 'bg-blue-500/10 text-blue-400',
    Botnet: 'bg-amber-500/10 text-amber-400',
    DoS: 'bg-purple-500/10 text-purple-400',
    BruteForce: 'bg-orange-500/10 text-orange-400',
    'Web attack': 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
    'Web Attack': 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
    Benign: 'bg-emerald-500/10 text-emerald-400',
  };
  const cls = map[label] ?? 'bg-slate-700/50 text-slate-400';
  return { bg: cls, text: '' };
}

// ─── Props Interface ──────────────────────────────────────────────────────────
interface ThreatAnalyticsProps {
  flows: FlowUIModel[];
  aggregation: LabelAggregation[];
  topAttackers: TopAttacker[];
  timeline: TimelinePoint[];
  queryParams: FlowQueryParams;
  isLoading: boolean;
  error: string | null;
  feedbackSuccess: string | null;
  availableHomes: string[];         // home IDs đã biết – dùng cho @ autocomplete
  onQueryParamsChange: (params: FlowQueryParams) => void;
  onSearch: () => void;
  onSubmitFeedback: (req: LabelFeedbackRequest) => void;
}

// ─── Time Range Presets ───────────────────────────────────────────────────────
type TimePreset = '24h' | '7d' | 'custom';

// ─── Relabel Modal ────────────────────────────────────────────────────────────
const KNOWN_LABELS = ['PortScan', 'DDoS', 'Botnet', 'DoS', 'BruteForce', 'Web Attack', 'Benign'];

interface RelabelModalProps {
  flow: FlowUIModel | null;
  onClose: () => void;
  onSubmit: (req: LabelFeedbackRequest) => void;
}

const RelabelModal: React.FC<RelabelModalProps> = ({ flow, onClose, onSubmit }) => {
  const [selected, setSelected] = useState('');
  const [note, setNote] = useState('');

  useEffect(() => {
    if (flow) {
      setSelected(flow.trueLabel ?? flow.predictedLabel);
      setNote('');
    }
  }, [flow]);

  if (!flow) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="bg-slate-900 border border-slate-700 rounded-2xl p-6 w-full max-w-sm shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="text-base font-bold text-slate-100 mb-1">Relabel Flow</h3>
        <p className="text-xs text-slate-500 font-mono mb-4">
          ID: {flow.id.slice(0, 16)}…
        </p>

        <div className="mb-4">
          <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-2">
            AI Predicted
          </p>
          <span
            className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${labelColor(flow.predictedLabel).bg}`}
          >
            {flow.predictedLabel}
          </span>
        </div>

        <div className="mb-4">
          <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-2">
            True Label
          </p>
          <div className="grid grid-cols-2 gap-2">
            {KNOWN_LABELS.map((lbl) => (
              <button
                key={lbl}
                onClick={() => setSelected(lbl)}
                className={`px-3 py-2 rounded-lg text-xs font-bold border transition-all ${selected === lbl
                    ? 'border-blue-500 bg-blue-500/20 text-blue-300'
                    : 'border-slate-700 bg-slate-800 text-slate-400 hover:border-slate-600'
                  }`}
              >
                {lbl}
              </button>
            ))}
          </div>
        </div>

        <div className="mb-5">
          <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-2">
            Note (optional)
          </p>
          <textarea
            value={note}
            onChange={(e) => setNote(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-300 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500/50"
            rows={2}
            placeholder="Reason for relabeling…"
          />
        </div>

        <div className="flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 py-2 rounded-lg border border-slate-700 text-xs font-bold text-slate-400 hover:border-slate-600 transition-all"
          >
            Cancel
          </button>
          <button
            onClick={() => {
              if (selected) {
                onSubmit({ flowId: flow.id, trueLabel: selected, note });
                onClose();
              }
            }}
            disabled={!selected}
            className="flex-1 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-xs font-bold text-white transition-all disabled:opacity-50"
          >
            Submit Feedback
          </button>
        </div>
      </motion.div>
    </div>
  );
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-[#0f172a] border border-slate-700 p-3 rounded-lg shadow-2xl outline-none">
        {label && <p className="text-slate-400 text-xs font-bold mb-2">{label}</p>}
        {payload.map((entry: any, index: number) => (
          <p
            key={index}
            className="text-sm font-medium"
            style={{ color: entry.color || entry.payload?.fill || '#f1f5f9' }}
          >
            {entry.name} : {entry.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

// ─── Main Component ───────────────────────────────────────────────────────────
export const ThreatAnalytics: React.FC<ThreatAnalyticsProps> = ({
  flows,
  aggregation,
  topAttackers,
  timeline,
  queryParams,
  isLoading,
  error,
  feedbackSuccess,
  availableHomes,
  onQueryParamsChange,
  onSearch,
  onSubmitFeedback,
}) => {
  const [activePreset, setActivePreset] = useState<TimePreset>('7d');
  const [localQuery, setLocalQuery] = useState(queryParams.query);
  const [relabelTarget, setRelabelTarget] = useState<FlowUIModel | null>(null);
  const [showAllAttackers, setShowAllAttackers] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const { t } = useLanguage();

  // ─── @ Syntax Autocomplete ──────────────────────────────────────────────────
  // Cú pháp: "DDoS" | "@home1" | "DDoS @home1"
  const AUTOCOMPLETE_LABELS = ['PortScan', 'DDoS', 'DoS', 'Botnet', 'BruteForce', 'Web Attack', 'Benign'];

  const ghostSuggestion = useCallback((q: string): string => {
    if (!q.trim()) return '';

    // ── @ prefix → autocomplete home name (hỗ trợ tên có dấu cách) ────
    const atIdx = q.lastIndexOf('@');
    if (atIdx >= 0) {
      const partial = q.slice(atIdx + 1).toLowerCase();
      if (!partial) return ''; // chỉ gõ '@', chưa có gì → không ghost
      const match = availableHomes.find(
        (h) => h.toLowerCase().startsWith(partial) && h.toLowerCase() !== partial
      );
      return match ? match.slice(partial.length) : '';
    }

    // ── Plain text → autocomplete label ────────────────────────────────
    const lastSpace = q.lastIndexOf(' ');
    const lastWord  = lastSpace >= 0 ? q.slice(lastSpace + 1) : q;
    if (!lastWord) return '';

    const lower = lastWord.toLowerCase();
    const match = AUTOCOMPLETE_LABELS.find(
      (lbl) => lbl.toLowerCase().startsWith(lower) && lbl.toLowerCase() !== lower
    );
    return match ? match.slice(lastWord.length) : '';
  }, [availableHomes]);

  const [ghost, setGhost] = useState('');
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Pagination logic
  useEffect(() => {
    setCurrentPage(1);
  }, [flows]);

  const rowsPerPage = 10;
  const totalPages = Math.ceil(flows.length / rowsPerPage);
  const paginatedFlows = flows.slice((currentPage - 1) * rowsPerPage, currentPage * rowsPerPage);

  // Unique labels in timeline for BarChart keys
  const timelineLabels = Array.from(
    new Set(timeline.flatMap((pt) => Object.keys(pt).filter((k) => k !== 'time')))
  );

  function applyPreset(preset: TimePreset) {
    setActivePreset(preset);
    const now = new Date().toISOString();
    if (preset === '24h') {
      onQueryParamsChange({
        ...queryParams,
        from: new Date(Date.now() - 86400_000).toISOString(),
        to: now,
      });
    } else if (preset === '7d') {
      onQueryParamsChange({
        ...queryParams,
        from: new Date(Date.now() - 7 * 86400_000).toISOString(),
        to: now,
      });
    }
  }

  function handleSearch() {
    onQueryParamsChange({ ...queryParams, query: localQuery });
    setGhost('');
    // Give React one tick to flush the state, then fetch
    setTimeout(() => onSearch(), 0);
  }

  function handleSuggest(val: string) {
    setLocalQuery(val);
    setGhost('');
    onQueryParamsChange({ ...queryParams, query: val });
    setTimeout(() => onSearch(), 0);
  }

  function handleQueryChange(val: string) {
    setLocalQuery(val);
    setGhost(ghostSuggestion(val));
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Tab' && ghost) {
      e.preventDefault();
      // Append ghost text trực tiếp (hỗ trợ multi-word home name)
      setLocalQuery(localQuery + ghost);
      setGhost('');
    } else if (e.key === 'Enter') {
      handleSearch();
    } else if (e.key === 'Escape') {
      setGhost('');
    }
  }

  function handleExportCsv() {
    if (!flows.length) return;
    const header =
      'Timestamp,Src_IP,Dst_Port,Duration,In_Bytes,Out_Bytes,Flags,Predicted_Label,User_Confirmed_Label,Confidence\n';
    const rows = flows
      .map((f) =>
        [
          f.time,
          f.srcIp,
          f.dstPort,
          f.duration,
          f.inBytes,
          f.outBytes,
          `"${f.tcpFlags}"`,
          f.predictedLabel,
          f.trueLabel ?? 'Pending',
          f.confidencePct,
        ].join(',')
      )
      .join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `threat-logs-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // Pie chart data – dùng aggregation từ DB (hoặc empty chart placeholder)
  const pieData = aggregation.length
    ? aggregation.map((a) => ({ name: a.label, value: a.count, color: a.color }))
    : [{ name: 'No Data', value: 1, color: '#1e293b' }];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-slate-100">
            {t('threatAnalytics', 'title')}
          </h2>
          <p className="text-slate-400 text-sm">
            {t('threatAnalytics', 'subTitle')}
          </p>
        </div>
        <div className="flex items-center gap-3 bg-slate-900 border border-slate-800 p-1 rounded-lg">
          <button
            onClick={() => { applyPreset('24h'); setTimeout(() => onSearch(), 0); }}
            className={`px-3 py-1.5 text-xs font-bold rounded-md transition-all ${activePreset === '24h'
                ? 'bg-slate-800 text-slate-200'
                : 'text-slate-500 hover:text-slate-300'
              }`}
          >
            {t('threatAnalytics', 'last24h') || 'Last 24 hours'}
          </button>
          <button
            onClick={() => { applyPreset('7d'); setTimeout(() => onSearch(), 0); }}
            className={`px-3 py-1.5 text-xs font-bold rounded-md transition-all ${activePreset === '7d'
                ? 'bg-slate-800 text-slate-200'
                : 'text-slate-500 hover:text-slate-300'
              }`}
          >
            {t('threatAnalytics', 'last7d') || 'Last 7 days'}
          </button>
          <button
            onClick={() => setActivePreset('custom')}
            className={`px-3 py-1.5 text-xs font-bold rounded-md transition-all ${activePreset === 'custom'
                ? 'bg-slate-800 text-slate-200'
                : 'text-slate-500 hover:text-slate-300'
              }`}
          >
            {t('threatAnalytics', 'customRange') || 'Custom Range'}
          </button>
          <div className="w-px h-4 bg-slate-800 mx-1" />
          <button
            onClick={() => onSearch()}
            disabled={isLoading}
            className="p-1.5 text-slate-400 hover:text-slate-200 disabled:opacity-50"
            title="Refresh"
          >
            <RefreshCw size={16} className={isLoading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Custom Range Inputs */}
      {activePreset === 'custom' && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="flex gap-4 items-center bg-slate-900 border border-slate-800 rounded-xl px-4 py-3"
        >
          <span className="text-xs text-slate-500 uppercase tracking-widest font-bold">From</span>
          <input
            type="datetime-local"
            value={queryParams.from.slice(0, 16)}
            onChange={(e) =>
              onQueryParamsChange({ ...queryParams, from: new Date(e.target.value).toISOString() })
            }
            className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500/50 [&::-webkit-calendar-picker-indicator]:invert"
          />
          <span className="text-xs text-slate-500 uppercase tracking-widest font-bold">To</span>
          <input
            type="datetime-local"
            value={queryParams.to.slice(0, 16)}
            onChange={(e) =>
              onQueryParamsChange({ ...queryParams, to: new Date(e.target.value).toISOString() })
            }
            className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500/50 [&::-webkit-calendar-picker-indicator]:invert"
          />
          <button
            onClick={() => onSearch()}
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 rounded-lg text-xs font-bold text-white transition-all"
          >
            Apply
          </button>
        </motion.div>
      )}

      {/* Feedback Toast */}
      <AnimatePresence>
        {feedbackSuccess && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex items-center gap-2 bg-emerald-900/30 border border-emerald-500/50 rounded-xl px-4 py-3 text-sm text-emerald-400"
          >
            <CheckCircle2 size={16} />
            {feedbackSuccess}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error Banner */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center gap-2 bg-red-900/20 border border-red-500/50 rounded-xl px-4 py-3 text-sm text-red-400"
          >
            <AlertCircle size={16} />
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Search Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl">
        <div className="flex gap-3">
          {/* Input wrapper – bg here so ghost overlay is visible */}
          <div className="flex-1 relative group bg-slate-950 border border-slate-800 rounded-lg focus-within:ring-2 focus-within:ring-blue-500/50 focus-within:border-blue-500 transition-all">
            {/* SPL badge */}
            <div className="absolute left-4 top-1/2 -translate-y-1/2 flex items-center gap-2 pointer-events-none z-20">
              <span className="text-blue-500 font-mono text-sm font-bold">SPL</span>
              <div className="w-px h-4 bg-slate-700" />
            </div>
            {/* Ghost overlay */}
            {ghost && (
              <div
                aria-hidden="true"
                className="absolute inset-0 pl-16 pr-4 py-3 font-mono text-sm pointer-events-none flex items-center overflow-hidden z-10"
              >
                <span className="invisible whitespace-pre">{localQuery}</span>
                <span className="text-slate-500 whitespace-pre">{ghost}</span>
                <span className="ml-1 text-[9px] text-slate-600 font-sans self-center border border-slate-700 rounded px-1 py-0.5 leading-none">
                  Tab
                </span>
              </div>
            )}

            <input
              ref={searchInputRef}
              type="text"
              value={localQuery}
              onChange={(e) => handleQueryChange(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder='DDoS  ·  @HomeName  ·  DDoS @HomeName'
              className="relative z-10 w-full bg-transparent pl-16 pr-10 py-3 font-mono text-sm text-slate-300 focus:outline-none rounded-lg placeholder:text-slate-700"
              autoComplete="off"
              spellCheck={false}
              maxLength={200}
            />

            {/* Clear button */}
            {localQuery && (
              <button
                onClick={() => {
                  setLocalQuery('');
                  setGhost('');
                  onQueryParamsChange({ ...queryParams, query: '' });
                  setTimeout(() => onSearch(), 0);
                  searchInputRef.current?.focus();
                }}
                className="absolute right-3 top-1/2 -translate-y-1/2 z-20 p-1 text-slate-600 hover:text-slate-300 hover:bg-slate-800 rounded-md transition-all"
                title="Clear search"
              >
                <X size={14} />
              </button>
            )}
          </div>
          <button
            onClick={handleSearch}
            disabled={isLoading}
            className="bg-blue-600 hover:bg-blue-500 disabled:opacity-60 text-white px-6 py-3 rounded-lg font-bold flex items-center gap-2 transition-all shadow-lg shadow-blue-600/20"
          >
            <Search size={18} />
            Search
          </button>
        </div>

        {/* Syntax hint + quick label chips */}
        <div className="mt-3 px-1 space-y-1.5">
          {/* Syntax guide */}
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-[10px] text-slate-600 uppercase tracking-widest font-bold shrink-0">Syntax:</span>
            {[
              { ex: 'DDoS',             desc: 'by label' },
              { ex: '@HomeName',         desc: 'by home name' },
              { ex: 'DDoS @HomeName',    desc: 'label + home' },
            ].map(({ ex, desc }) => (
              <span key={ex} className="flex items-center gap-1">
                <code
                  onClick={() => handleSuggest(ex)}
                  className="text-[10px] font-mono text-slate-400 bg-slate-800 px-1.5 py-0.5 rounded cursor-pointer hover:bg-slate-700 transition-colors"
                >
                  {ex}
                </code>
                <span className="text-[10px] text-slate-600">{desc}</span>
              </span>
            ))}
          </div>

          {/* Quick label chips */}
          <div className="flex items-center gap-x-3 gap-y-1 flex-wrap">
            <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold shrink-0">Labels:</span>
            {[
              { label: 'PortScan',   color: 'text-rose-400' },
              { label: 'DDoS',       color: 'text-blue-400' },
              { label: 'DoS',        color: 'text-purple-400' },
              { label: 'Botnet',     color: 'text-amber-400' },
              { label: 'BruteForce', color: 'text-orange-400' },
              { label: 'Web Attack', color: 'text-cyan-400' },
              { label: 'Benign',     color: 'text-emerald-400' },
            ].map(({ label, color }) => (
              <button
                key={label}
                onClick={() => handleSuggest(label)}
                className={`text-[10px] font-mono hover:underline transition-colors ${color} ${
                  localQuery === label ? 'underline font-bold' : ''
                }`}
              >
                {label}
              </button>
            ))}
            <span className="text-slate-800 text-[10px] select-none">|</span>
            <button
              onClick={() => handleSuggest('')}
              className="text-[10px] text-slate-500 hover:text-slate-300 font-mono hover:underline transition-colors"
            >
              All
            </button>
          </div>

          {/* Available home chips (populated after first data load) */}
          {availableHomes.length > 0 && (
            <div className="flex items-center gap-x-3 gap-y-1 flex-wrap">
              <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold shrink-0">Homes:</span>
              {availableHomes.map((h) => {
                const token = `@${h}`;
                return (
                  <button
                    key={h}
                    onClick={() => handleSuggest(token)}
                    className={`text-[10px] font-mono text-sky-400 hover:underline transition-colors ${
                      localQuery === token ? 'underline font-bold' : ''
                    }`}
                  >
                    {token}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Analytics Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Attack Distribution – Pie */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl flex flex-col">
          <div className="flex items-center gap-2 mb-6">
            <PieChartIcon size={18} className="text-blue-400" />
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">
              Attack Distribution
            </h3>
            {isLoading && (
              <RefreshCw size={12} className="text-slate-600 animate-spin ml-auto" />
            )}
          </div>
          <div className="flex-1 min-h-[240px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius="60%"
                  outerRadius="80%"
                  paddingAngle={pieData.length === 1 ? 0 : 5}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend verticalAlign="bottom" height={36} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Attack Timeline – Stacked Bar */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl flex flex-col">
          <div className="flex items-center gap-2 mb-6">
            <BarChart3 size={18} className="text-blue-400" />
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">
              Attack Timeline (Stacked)
            </h3>
          </div>
          <div className="flex-1 min-h-[240px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={timeline.length ? timeline : [{ time: 'No Data' }]}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis
                  dataKey="time"
                  stroke="#64748b"
                  fontSize={10}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  stroke="#64748b"
                  fontSize={10}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip content={<CustomTooltip />} />
                {timelineLabels.map((lbl) => (
                  <Bar
                    key={lbl}
                    dataKey={lbl}
                    stackId="a"
                    fill={LABEL_COLOR[String(lbl)] ?? DEFAULT_LABEL_COLOR}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top Attacker IPs */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl flex flex-col">
          <div className="flex items-center gap-2 mb-6">
            <Filter size={18} className="text-blue-400" />
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">
              Top Attacker IPs
            </h3>
          </div>
          <div className="flex-1 overflow-y-auto pr-2 space-y-3">
            {topAttackers.length === 0 && !isLoading ? (
              <p className="text-xs text-slate-600 text-center py-6">No data</p>
            ) : null}
            {topAttackers.slice(0, 5).map((attacker, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-3 bg-slate-950/50 border border-slate-800 rounded-lg hover:border-slate-700 transition-colors"
              >
                <div>
                  <p className="text-sm font-mono font-bold text-slate-200">
                    {attacker.ip}
                  </p>
                  <p className="text-[10px] text-slate-500 uppercase tracking-widest">
                    {attacker.label}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-sm font-bold text-rose-400">{attacker.count}</p>
                  <p className="text-[10px] text-slate-500 uppercase tracking-widest">
                    Events
                  </p>
                </div>
              </div>
            ))}
          </div>
          {topAttackers.length > 5 && (
            <button 
              onClick={() => setShowAllAttackers(true)}
              className="w-full mt-4 py-2 text-xs font-bold text-blue-400 hover:text-blue-300 transition-colors flex items-center justify-center gap-2"
            >
              View Full List <ExternalLink size={12} />
            </button>
          )}
        </div>
      </div>

      {/* Raw Data Grid */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl shadow-xl overflow-hidden">
        <div className="p-6 border-b border-slate-800 flex justify-between items-center bg-slate-900/50">
          <div>
            <h3 className="text-lg font-bold text-slate-100">{t('threatAnalytics', 'rawLogs') || 'Raw Network Logs'}</h3>
            <p className="text-xs text-slate-500">
              Human-in-the-loop verification for AI model training
              {flows.length > 0 && (
                <span className="ml-2 text-slate-600">({flows.length} records)</span>
              )}
            </p>
          </div>
          <button
            onClick={handleExportCsv}
            disabled={flows.length === 0}
            className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 rounded-lg text-xs font-bold text-slate-300 transition-all"
          >
            <Download size={14} />
            Export CSV
          </button>
        </div>
        <div className="overflow-x-auto custom-scrollbar">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-950/50 border-b border-slate-800">
                <th className="px-6 py-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                  {t('threatAnalytics', 'tblTimestamp')}
                </th>
                <th className="px-6 py-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                  Src_IP
                </th>
                <th className="px-6 py-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                  Dst_Port
                </th>
                <th className="px-6 py-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                  Duration
                </th>
                <th className="px-6 py-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                  In_Bytes
                </th>
                <th className="px-6 py-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                  Out_Bytes
                </th>
                <th className="px-6 py-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                  Flags
                </th>
                <th className="px-6 py-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                  Predicted_Label
                </th>
                <th className="px-6 py-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest text-center">
                  {t('threatAnalytics', 'tblStatus') || 'Actions'}
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {/* Loading skeleton rows */}
              {isLoading && flows.length === 0 &&
                [...Array(5)].map((_, i) => (
                  <tr key={i} className="animate-pulse">
                    {[...Array(9)].map((_, j) => (
                      <td key={j} className="px-6 py-4">
                        <div className="h-3 bg-slate-800 rounded w-3/4" />
                      </td>
                    ))}
                  </tr>
                ))
              }

              {/* Empty state */}
              {!isLoading && flows.length === 0 && (
                <tr>
                  <td
                    colSpan={9}
                    className="px-6 py-12 text-center text-sm text-slate-600"
                  >
                    No logs found. Try adjusting the time range or search query.
                  </td>
                </tr>
              )}

              {/* Real data rows */}
              {paginatedFlows.map((flow) => (
                <tr
                  key={flow.id}
                  className="hover:bg-slate-800/30 transition-colors group"
                >
                  <td className="px-6 py-4 text-xs font-mono text-slate-400">
                    {flow.time}
                  </td>
                  <td className="px-6 py-4 text-xs font-mono text-slate-200">
                    {flow.srcIp}
                  </td>
                  <td className="px-6 py-4 text-xs font-mono text-slate-400">
                    {flow.dstPort}
                  </td>
                  <td className="px-6 py-4 text-xs font-mono text-slate-400">
                    {flow.duration}
                  </td>
                  <td className="px-6 py-4 text-xs font-mono text-slate-400">
                    {flow.inBytes}
                  </td>
                  <td className="px-6 py-4 text-xs font-mono text-slate-400">
                    {flow.outBytes}
                  </td>
                  <td className="px-6 py-4 text-xs font-mono text-slate-400">
                    {flow.tcpFlags}
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex flex-col gap-1">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${labelColor(flow.predictedLabel).bg
                          }`}
                      >
                        {flow.predictedLabel}
                      </span>
                      {flow.trueLabel && (
                        <span
                          className={`text-[10px] font-bold uppercase mt-0.5 ${flow.trueLabel === flow.predictedLabel
                              ? 'text-emerald-500'
                              : 'text-rose-500'
                            }`}
                        >
                          {flow.trueLabel === flow.predictedLabel ? '✓' : '↳'}{' '}
                          {flow.trueLabel}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    {flow.hasFeedback ? (
                      <div className="flex items-center justify-center">
                        <span className={`text-[9px] px-2 py-1 rounded font-bold uppercase ${flow.trueLabel === flow.predictedLabel
                            ? 'bg-emerald-500/10 text-emerald-400'
                            : 'bg-orange-500/10 text-orange-400'
                          }`}>
                          {flow.trueLabel === flow.predictedLabel ? '✓ Accepted' : '↳ Corrected'}
                        </span>
                      </div>
                    ) : (
                      <div className="flex items-center justify-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                        {/* Confirm Label (False Positive resolve) */}
                        <button
                          onClick={() => {
                            onSubmitFeedback({
                              flowId: flow.id,
                              trueLabel: flow.predictedLabel,
                            });
                          }}
                          className="p-1.5 bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 rounded-md transition-all"
                          title="Confirm Label"
                        >
                          <CheckCircle2 size={14} />
                        </button>
                        {/* Relabel (open modal) */}
                        <button
                          onClick={() => setRelabelTarget(flow)}
                          className="p-1.5 bg-slate-800 text-slate-500 hover:bg-slate-700 hover:text-slate-300 rounded-md transition-all"
                          title="Relabel / False Positive"
                        >
                          <XCircle size={14} />
                        </button>
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Pagination Controls */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between px-6 py-4 border-t border-slate-800 bg-slate-900/50">
              <span className="text-xs text-slate-500">
                Showing {(currentPage - 1) * rowsPerPage + 1} to {Math.min(currentPage * rowsPerPage, flows.length)} of {flows.length} entries
              </span>
              <div className="flex gap-2">
                <button 
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 disabled:hover:bg-slate-800 rounded-md text-xs font-bold text-slate-300 transition-colors"
                >
                  Prev
                </button>
                <div className="flex gap-1">
                  {[...Array(totalPages)].map((_, idx) => (
                    <button
                      key={idx}
                      onClick={() => setCurrentPage(idx + 1)}
                      className={`w-7 h-7 rounded-md flex items-center justify-center text-xs font-bold transition-colors ${currentPage === idx + 1 ? 'bg-blue-600 text-white' : 'bg-slate-800 hover:bg-slate-700 text-slate-300'}`}
                    >
                      {idx + 1}
                    </button>
                  ))}
                </div>
                <button 
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                  className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 disabled:hover:bg-slate-800 rounded-md text-xs font-bold text-slate-300 transition-colors"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Relabel Modal */}
      <AnimatePresence>
        {relabelTarget && (
          <RelabelModal
            flow={relabelTarget}
            onClose={() => setRelabelTarget(null)}
            onSubmit={onSubmitFeedback}
          />
        )}
      </AnimatePresence>

      {/* TOP ATTACKERS FULL LIST MODAL */}
      <AnimatePresence>
        {showAllAttackers && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="bg-slate-900 border border-slate-700 rounded-xl max-w-lg w-full flex flex-col max-h-[85vh] shadow-2xl overflow-hidden"
            >
              <div className="p-5 border-b border-slate-800 flex justify-between items-center bg-slate-950/50">
                <div>
                  <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                    <Filter size={18} className="text-blue-400" />
                    All Attacker IPs
                  </h2>
                  <p className="text-xs text-slate-500 mt-1">
                    Showing all {topAttackers.length} IP sources detected in current scope.
                  </p>
                </div>
                <button
                  onClick={() => setShowAllAttackers(false)}
                  className="p-1 hover:bg-slate-800 rounded-full text-slate-400 transition-colors"
                >
                  <X size={18} />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-5 space-y-3 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-slate-900/50 [&::-webkit-scrollbar-thumb]:bg-slate-700 [&::-webkit-scrollbar-thumb]:rounded-full hover:[&::-webkit-scrollbar-thumb]:bg-slate-600">
                {topAttackers.map((attacker, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between p-3 bg-slate-950 border border-slate-800 rounded-lg"
                  >
                    <div className="flex justify-between items-center w-full">
                      <div>
                        <p className="text-sm font-mono font-bold text-slate-200">
                          {attacker.ip}
                        </p>
                        <p className="text-[10px] text-slate-500 uppercase tracking-widest mt-1">
                          {attacker.label}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-base font-bold text-rose-400">
                          {attacker.count}
                        </p>
                        <p className="text-[10px] text-slate-500 uppercase tracking-widest">
                          Events
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="p-4 border-t border-slate-800 bg-slate-950/50">
                <button
                  onClick={() => setShowAllAttackers(false)}
                  className="w-full py-2 bg-slate-800 hover:bg-slate-700 text-sm font-bold text-slate-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
};
