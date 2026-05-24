import React from 'react';
import { motion } from 'motion/react';
import { Activity, Zap, WifiOff } from 'lucide-react';
import type { EdgeNodeUIModel } from '../../model/types';

interface FleetHealthPanelProps {
  nodes: EdgeNodeUIModel[];
  isLoading: boolean;
}

const StatRow: React.FC<{ label: string; value: React.ReactNode; sub?: string }> = ({ label, value, sub }) => (
  <div className="flex justify-between items-center py-2">
    <span className="text-xs text-slate-500">{label}</span>
    <div className="text-right">
      <div className="text-sm font-bold">{value}</div>
      {sub && <div className="text-[10px] text-slate-500">{sub}</div>}
    </div>
  </div>
);

export const FleetHealthPanel: React.FC<FleetHealthPanelProps> = ({ nodes, isLoading }) => {
  const total = nodes.length;
  const online = nodes.filter(n => n.status === 'online').length;
  const offline = nodes.filter(n => n.status === 'offline').length;
  const onlinePct = total > 0 ? Math.round((online / total) * 100) : 0;

  // Avg metrics (online nodes only)
  const onlineNodes = nodes.filter(n => n.status === 'online' && n.cpuRaw != null);
  const avgCpu = onlineNodes.length > 0
    ? onlineNodes.reduce((s, n) => s + (n.cpuRaw ?? 0), 0) / onlineNodes.length
    : null;
  const avgRam = onlineNodes.length > 0
    ? onlineNodes.reduce((s, n) => s + (n.memRaw ?? 0), 0) / onlineNodes.length
    : null;

  if (isLoading) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 animate-pulse space-y-4">
        <div className="h-5 bg-slate-800 rounded w-24" />
        {[...Array(5)].map((_, i) => <div key={i} className="h-4 bg-slate-800 rounded" />)}
      </div>
    );
  }

  return (
    <div className="bg-slate-900/80 backdrop-blur-sm border border-slate-800 rounded-xl p-5 shadow-xl">
      <div className="flex items-center gap-2 mb-4">
        <Activity size={16} className="text-blue-400" />
        <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">Fleet Health</h3>
      </div>

      <div className="divide-y divide-slate-800/60">
        <StatRow label="Total Nodes" value={<span className="text-slate-200">{total}</span>} />
        <StatRow
          label="Online"
          value={<span className="text-emerald-400">{online}</span>}
          sub={total > 0 ? `${onlinePct}% uptime` : undefined}
        />
        <StatRow label="Offline" value={<span className={offline > 0 ? 'text-rose-400' : 'text-slate-500'}>{offline}</span>} />
        {avgCpu != null && <StatRow label="Avg CPU" value={<span className="text-blue-400">{avgCpu.toFixed(1)}%</span>} />}
        {avgRam != null && <StatRow label="Avg RAM" value={<span className="text-purple-400">{avgRam.toFixed(1)}%</span>} />}
      </div>

      {/* Overall health bar */}
      <div className="mt-4">
        <div className="flex justify-between text-[10px] text-slate-500 mb-1">
          <span>Health Score</span>
          <span>{onlinePct}%</span>
        </div>
        <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
          <motion.div
            className={`h-full rounded-full ${
              onlinePct >= 90 ? 'bg-emerald-500' :
              onlinePct >= 70 ? 'bg-amber-500' : 'bg-rose-500'
            }`}
            initial={{ width: 0 }}
            animate={{ width: `${onlinePct}%` }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
          />
        </div>
      </div>

      {/* Status indicators */}
      {offline > 0 && (
        <div className="mt-3 flex items-center gap-2 text-[11px] text-rose-400 bg-rose-500/10 border border-rose-500/20 rounded-lg px-3 py-2">
          <WifiOff size={12} />
          <span>{offline} node{offline > 1 ? 's' : ''} offline</span>
        </div>
      )}
      {offline === 0 && total > 0 && (
        <div className="mt-3 flex items-center gap-2 text-[11px] text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 rounded-lg px-3 py-2">
          <Zap size={12} />
          <span>All systems operational</span>
        </div>
      )}
    </div>
  );
};
