import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, Loader2, Server, Bell, CornerDownLeft } from 'lucide-react';
import { runGlobalSearch, type GlobalSearchAlert, type GlobalSearchNode } from '../search/runGlobalSearch';

type FlatItem =
  | { kind: 'node'; data: GlobalSearchNode }
  | { kind: 'alert'; data: GlobalSearchAlert };

export const GlobalSearchBar: React.FC = () => {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  const [q, setQ] = useState('');
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [nodes, setNodes] = useState<GlobalSearchNode[]>([]);
  const [alerts, setAlerts] = useState<GlobalSearchAlert[]>([]);
  const [activeIdx, setActiveIdx] = useState(0);

  const flat: FlatItem[] = [
    ...nodes.map((data) => ({ kind: 'node' as const, data })),
    ...alerts.map((data) => ({ kind: 'alert' as const, data })),
  ];

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        inputRef.current?.focus();
        setOpen(true);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, []);

  useEffect(() => {
    const t = q.trim();
    if (t.length < 2) {
      setNodes([]);
      setAlerts([]);
      setLoading(false);
      return;
    }

    setLoading(true);
    const id = window.setTimeout(() => {
      void runGlobalSearch(t).then(({ nodes: n, alerts: a }) => {
        setNodes(n);
        setAlerts(a);
        setLoading(false);
        setActiveIdx(0);
      });
    }, 280);

    return () => {
      window.clearTimeout(id);
    };
  }, [q]);

  const goToItem = useCallback(
    (item: FlatItem) => {
      setOpen(false);
      setQ('');
      if (item.kind === 'node') {
        navigate(`/fleet?nodeId=${encodeURIComponent(item.data.id)}`);
      } else {
        navigate(`/alerts?alertId=${encodeURIComponent(item.data.alert_id)}`);
      }
    },
    [navigate],
  );

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (flat.length === 0) return;
    goToItem(flat[activeIdx] ?? flat[0]);
  };

  const onKeyDownInput = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setOpen(false);
      return;
    }
    if (!open || flat.length === 0) return;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIdx((i) => Math.min(i + 1, flat.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === 'Enter' && flat.length > 0) {
      e.preventDefault();
      goToItem(flat[activeIdx] ?? flat[0]);
    }
  };

  const showPanel = open && q.trim().length >= 2;

  return (
    <div ref={wrapRef} className="relative group w-full max-w-md">
      <form
        className="relative"
        onSubmit={onSubmit}
        onFocus={() => setOpen(true)}
      >
        <Search
          className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-blue-400 transition-colors pointer-events-none"
          size={16}
        />
        <input
          ref={inputRef}
          type="search"
          value={q}
          onChange={(e) => {
            setQ(e.target.value);
            setOpen(true);
          }}
          onKeyDown={onKeyDownInput}
          placeholder="Search nodes, IPs, or threats..."
          autoComplete="off"
          className="bg-slate-900 border border-slate-800 rounded-lg pl-10 pr-16 py-2 text-sm text-slate-200 w-full focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-all shadow-inner"
        />
        <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-0.5 opacity-50 select-none pointer-events-none">
          <kbd className="hidden sm:inline-flex px-1.5 py-0.5 text-[10px] items-center justify-center font-mono font-bold text-slate-400 bg-slate-800 border border-slate-700 rounded shadow-sm">
            Ctrl
          </kbd>
          <span className="text-slate-500 text-xs hidden sm:inline">+</span>
          <kbd className="hidden sm:inline-flex px-1.5 py-0.5 text-[10px] items-center justify-center font-mono font-bold text-slate-400 bg-slate-800 border border-slate-700 rounded shadow-sm">
            K
          </kbd>
        </div>
      </form>

      {showPanel && (
        <div className="absolute left-0 right-0 top-full mt-1.5 z-[60] bg-slate-900 border border-slate-800 rounded-xl shadow-2xl overflow-hidden max-h-[min(70vh,420px)] flex flex-col">
          {loading && (
            <div className="flex items-center justify-center gap-2 py-8 text-slate-500 text-sm">
              <Loader2 className="animate-spin" size={18} />
              Searching…
            </div>
          )}

          {!loading && flat.length === 0 && (
            <div className="py-8 px-4 text-center text-slate-500 text-sm">No matches for this query.</div>
          )}

          {!loading && flat.length > 0 && (
            <ul className="overflow-y-auto custom-scrollbar py-2">
              {nodes.length > 0 && (
                <li className="px-3 pb-1">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-1">Edge nodes</p>
                </li>
              )}
              {nodes.map((n, i) => {
                const idx = i;
                const active = activeIdx === idx;
                return (
                  <li key={n.id}>
                    <button
                      type="button"
                      onMouseEnter={() => setActiveIdx(idx)}
                      onClick={() => goToItem({ kind: 'node', data: n })}
                      className={`w-full text-left px-3 py-2 flex items-start gap-3 transition-colors ${
                        active ? 'bg-blue-600/15 border-l-2 border-blue-500' : 'hover:bg-slate-800/80 border-l-2 border-transparent'
                      }`}
                    >
                      <Server size={16} className="text-blue-400 mt-0.5 shrink-0" />
                      <div className="min-w-0">
                        <p className="text-sm font-semibold text-slate-100 truncate">{n.node_code}</p>
                        <p className="text-xs text-slate-500 truncate">
                          {[n.ip_address, n.location_text].filter(Boolean).join(' · ') || '—'}
                        </p>
                      </div>
                    </button>
                  </li>
                );
              })}

              {alerts.length > 0 && (
                <li className="px-3 pt-2 pb-1 border-t border-slate-800/80 mt-1">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-1">Alerts</p>
                </li>
              )}
              {alerts.map((a, j) => {
                const idx = nodes.length + j;
                const active = activeIdx === idx;
                return (
                  <li key={a.alert_id}>
                    <button
                      type="button"
                      onMouseEnter={() => setActiveIdx(idx)}
                      onClick={() => goToItem({ kind: 'alert', data: a })}
                      className={`w-full text-left px-3 py-2 flex items-start gap-3 transition-colors ${
                        active ? 'bg-blue-600/15 border-l-2 border-blue-500' : 'hover:bg-slate-800/80 border-l-2 border-transparent'
                      }`}
                    >
                      <Bell size={16} className="text-amber-400 mt-0.5 shrink-0" />
                      <div className="min-w-0">
                        <p className="text-sm font-semibold text-slate-100 truncate">{a.alert_threat_type}</p>
                        <p className="text-xs text-slate-500 truncate">
                          {a.alert_source_ip || '—'} · {a.alert_severity} ·{' '}
                          {new Date(a.alert_created_at).toLocaleString()}
                        </p>
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}

          {!loading && q.trim().length >= 2 && (
            <div className="border-t border-slate-800 px-3 py-2 flex items-center justify-between text-[10px] text-slate-500 bg-slate-950/50">
              <span className="flex items-center gap-1">
                <CornerDownLeft size={12} /> Open
              </span>
              <span>↑↓ Navigate</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
