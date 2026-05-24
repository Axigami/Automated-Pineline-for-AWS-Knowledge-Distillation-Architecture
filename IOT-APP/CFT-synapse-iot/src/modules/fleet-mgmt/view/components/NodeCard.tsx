import React, { useState, useRef, useEffect } from 'react';
import { Cpu, Database, Wifi, Thermometer, RefreshCw, FileText, MoreVertical, Trash2, AlertTriangle, MapPin, Check } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer, Tooltip, YAxis } from 'recharts';
import { motion } from 'motion/react';
import { toast } from './Toast';
import type { EdgeNodeUIModel, TelemetryPoint } from '../../model/types';

const OFFLINE_SPARKLINE = Array.from({ length: 10 }, () => ({ value: 0 }));

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-800 border border-slate-700 px-2 py-1 rounded shadow-xl">
        <p className="text-[10px] font-bold text-slate-100">{payload[0].value.toFixed(1)}%</p>
      </div>
    );
  }
  return null;
};

const LatencyTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-800 border border-slate-700 px-2 py-1 rounded shadow-xl">
        <p className="text-[10px] font-bold text-slate-100">{payload[0].value.toFixed(0)}ms</p>
      </div>
    );
  }
  return null;
};

const TempTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-800 border border-slate-700 px-2 py-1 rounded shadow-xl">
        <p className="text-[10px] font-bold text-slate-100">{payload[0].value.toFixed(1)}°C</p>
      </div>
    );
  }
  return null;
};

interface NodeCardProps {
  node: EdgeNodeUIModel;
  telemetry?: TelemetryPoint[];
  onRestart?: (id: string) => void;
  onDelete?: (id: string) => Promise<void>;
  onEditLocation?: (id: string, newLocation: string) => Promise<void>;
  onClick?: () => void;
}

export const NodeCard: React.FC<NodeCardProps> = React.memo(({ node, telemetry = [], onRestart, onDelete, onEditLocation, onClick }) => {
  const [menuOpen, setMenuOpen] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [editingLocation, setEditingLocation] = useState(false);
  const [newLocation, setNewLocation] = useState(node.location);
  const [isSavingLocation, setIsSavingLocation] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    if (menuOpen) document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [menuOpen]);

  const handleDelete = async () => {
    if (!onDelete) return;
    setIsDeleting(true);
    try {
      await onDelete(node.id);
      toast('success', 'Node Deleted', `Successfully deleted node ${node.nodeCode}`);
    } catch (err: any) {
      toast('error', 'Deletion Failed', err.message);
    } finally {
      setIsDeleting(false);
      setConfirmDelete(false);
      setMenuOpen(false);
    }
  };

  const handleSaveLocation = async () => {
    if (!onEditLocation || !newLocation.trim()) return;
    setIsSavingLocation(true);
    try {
      await onEditLocation(node.id, newLocation.trim());
      setEditingLocation(false);
      toast('success', 'Location Updated', `Location for ${node.nodeCode} updated to ${newLocation.trim()}`);
    } catch (err: any) {
      toast('error', 'Update Failed', err.message);
    } finally {
      setIsSavingLocation(false);
    }
  };
  const isOffline = node.status === 'offline';

  const getSparklineData = (history: TelemetryPoint[], currentVal: number | null, key: keyof TelemetryPoint) => {
    if (isOffline) return OFFLINE_SPARKLINE;
    if (history.length > 1) return history.map(t => ({ value: t[key] as number }));
    const val = currentVal ?? 0;
    return Array.from({ length: 10 }, () => ({ value: val }));
  };

  const cpuSparkline = getSparklineData(telemetry, node.cpuRaw ?? 0, 'cpu');
  const ramSparkline = getSparklineData(telemetry, node.memRaw ?? 0, 'ram');
  const tempSparkline = getSparklineData(telemetry, node.tempRaw ?? 0, 'temp');
  const latencySparkline = getSparklineData(telemetry, node.latencyRaw ?? 0, 'latency');

  return (
    <motion.div
      onClick={onClick}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="bg-slate-900 border border-slate-800 rounded-xl p-5 hover:border-slate-700 transition-colors shadow-lg cursor-pointer"
    >
      <div className="flex justify-between items-start mb-6">
        <div className="flex items-center gap-3">
          <div
            className={`w-2 h-2 rounded-full ${
              node.status === 'offline'
                ? 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]'
                : 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]'
            }`}
          />
          <div>
            <h3 className="text-slate-100 text-lg font-bold tracking-tight">{node.nodeCode}</h3>
            <p className="text-slate-500 text-sm opacity-80">{node.location}</p>
          </div>
        </div>
        <div className="relative" ref={menuRef}>
          <button
            onClick={(e) => { e.stopPropagation(); setMenuOpen(!menuOpen); }}
            className="text-slate-500 hover:text-slate-300 transition-colors p-1 rounded hover:bg-slate-800"
          >
            <MoreVertical size={16} />
          </button>

          {/* Dropdown Menu */}
          {menuOpen && (
            <div className="absolute right-0 top-8 bg-slate-800 border border-slate-700 rounded-lg shadow-2xl z-50 min-w-[180px] overflow-hidden animate-in fade-in slide-in-from-top-1 duration-150">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setMenuOpen(false);
                  setNewLocation(node.location);
                  setEditingLocation(true);
                }}
                className="w-full flex items-center gap-2 px-4 py-2.5 text-sm text-slate-300 hover:bg-slate-700 transition-colors font-medium"
              >
                <MapPin size={14} /> Edit Location
              </button>
              <div className="border-t border-slate-700/50" />
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setMenuOpen(false);
                  setConfirmDelete(true);
                }}
                className="w-full flex items-center gap-2 px-4 py-2.5 text-sm text-red-400 hover:bg-red-500/10 transition-colors font-medium"
              >
                <Trash2 size={14} /> Delete Node
              </button>
            </div>
          )}
        </div>

        {/* Edit Location Modal */}
        {editingLocation && (
          <>
            <div className="fixed inset-0 bg-slate-950/70 backdrop-blur-sm z-[9998]" onClick={(e) => { e.stopPropagation(); setEditingLocation(false); }} />
            <div className="fixed inset-0 flex items-center justify-center z-[9999] pointer-events-none">
              <div className="bg-slate-900 border border-blue-500/30 rounded-2xl shadow-2xl p-6 max-w-sm w-full pointer-events-auto" onClick={(e) => e.stopPropagation()}>
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-full bg-blue-500/10 flex items-center justify-center">
                    <MapPin size={20} className="text-blue-400" />
                  </div>
                  <div>
                    <h4 className="text-slate-100 font-bold">Edit Location</h4>
                    <p className="text-slate-500 text-xs">Update location for <strong className="text-slate-300">{node.nodeCode}</strong></p>
                  </div>
                </div>
                <div className="space-y-1 mb-6">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">New Location</label>
                  <input
                    type="text"
                    value={newLocation}
                    onChange={(e) => setNewLocation(e.target.value)}
                    placeholder="E.g., Can Tho"
                    className="w-full bg-slate-950 border border-slate-700 text-slate-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all placeholder:text-slate-600"
                    autoFocus
                    onKeyDown={(e) => { if (e.key === 'Enter') handleSaveLocation(); }}
                  />
                  <p className="text-[10px] text-slate-500">The fleet map will auto-locate this position.</p>
                </div>
                <div className="flex justify-end gap-3">
                  <button
                    onClick={(e) => { e.stopPropagation(); setEditingLocation(false); }}
                    className="px-4 py-2 rounded-lg text-sm font-bold text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={(e) => { e.stopPropagation(); handleSaveLocation(); }}
                    disabled={isSavingLocation || !newLocation.trim()}
                    className="px-4 py-2 rounded-lg text-sm font-bold text-white bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-500/20 flex items-center gap-2"
                  >
                    <Check size={14} /> {isSavingLocation ? 'Saving...' : 'Save'}
                  </button>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Delete Confirmation Overlay */}
        {confirmDelete && (
          <>
            <div className="fixed inset-0 bg-slate-950/70 backdrop-blur-sm z-[9998]" onClick={(e) => { e.stopPropagation(); setConfirmDelete(false); }} />
            <div className="fixed inset-0 flex items-center justify-center z-[9999] pointer-events-none">
              <div className="bg-slate-900 border border-red-500/30 rounded-2xl shadow-2xl p-6 max-w-sm w-full pointer-events-auto" onClick={(e) => e.stopPropagation()}>
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-full bg-red-500/10 flex items-center justify-center">
                    <AlertTriangle size={20} className="text-red-500" />
                  </div>
                  <div>
                    <h4 className="text-slate-100 font-bold">Confirm Deletion</h4>
                    <p className="text-slate-500 text-xs">This action cannot be undone</p>
                  </div>
                </div>
                <p className="text-slate-300 text-sm mb-6">
                  Are you sure you want to delete node <strong className="text-slate-100">{node.nodeCode}</strong> and its associated Home?
                </p>
                <div className="flex justify-end gap-3">
                  <button
                    onClick={(e) => { e.stopPropagation(); setConfirmDelete(false); }}
                    className="px-4 py-2 rounded-lg text-sm font-bold text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDelete(); }}
                    disabled={isDeleting}
                    className="px-4 py-2 rounded-lg text-sm font-bold text-white bg-red-600 hover:bg-red-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-red-500/20"
                  >
                    {isDeleting ? 'Deleting...' : 'Delete Node'}
                  </button>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      <div className="grid grid-cols-4 gap-3 mb-6">
        <div className="space-y-1.5">
          <div className="flex items-center gap-1 text-[11px] text-slate-500 font-bold uppercase tracking-wider">
            <Cpu size={12} /> CPU
          </div>
          <p className="text-base font-bold text-slate-100 leading-none">{isOffline ? '0.0%' : node.cpuPct}</p>
          <div className="h-4 w-full">
            <ResponsiveContainer width="100%" height="100%" minWidth={0}>
              <LineChart data={cpuSparkline}>
                <YAxis hide domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} position={{ y: -20 }} isAnimationActive={false} />
                <Line type="monotone" dataKey="value" stroke={isOffline ? '#475569' : '#3b82f6'} strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="space-y-1.5">
          <div className="flex items-center gap-1 text-[11px] text-slate-500 font-bold uppercase tracking-wider">
            <Database size={12} /> RAM
          </div>
          <p className="text-base font-bold text-slate-100 leading-none">{isOffline ? '0.0%' : node.memPct}</p>
          <div className="h-4 w-full">
            <ResponsiveContainer width="100%" height="100%" minWidth={0}>
              <LineChart data={ramSparkline}>
                <YAxis hide domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} position={{ y: -20 }} isAnimationActive={false} />
                <Line type="monotone" dataKey="value" stroke={isOffline ? '#475569' : '#a855f7'} strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="space-y-1.5">
          <div className="flex items-center gap-1 text-[11px] text-slate-500 font-bold uppercase tracking-wider">
            <Thermometer size={12} /> TEMP
          </div>
          <p className="text-base font-bold text-slate-100 leading-none">{isOffline ? '0.0°C' : node.tempC}</p>
          <div className="h-4 w-full">
            <ResponsiveContainer width="100%" height="100%" minWidth={0}>
              <LineChart data={tempSparkline}>
                <YAxis hide domain={['auto', 'auto']} />
                <Tooltip content={<TempTooltip />} position={{ y: -20 }} isAnimationActive={false} />
                <Line type="monotone" dataKey="value" stroke={isOffline ? '#475569' : '#f59e0b'} strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="space-y-1.5">
          <div className="flex items-center gap-1 text-[11px] text-slate-500 font-bold uppercase tracking-wider">
            <Wifi size={12} /> LATENCY
          </div>
          <p className="text-base font-bold text-slate-100 leading-none">{isOffline ? '0ms' : node.latencyMs}</p>
          <div className="h-4 w-full">
            <ResponsiveContainer width="100%" height="100%" minWidth={0}>
              <LineChart data={latencySparkline}>
                <YAxis hide domain={['auto', 'auto']} />
                <Tooltip content={<LatencyTooltip />} position={{ y: -20 }} isAnimationActive={false} />
                <Line type="monotone" dataKey="value" stroke={isOffline ? '#475569' : '#10b981'} strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="space-y-3 pt-4 border-t border-slate-800/60">
        <div className="flex justify-between text-sm">
          <span className="text-slate-500 font-medium">Model Version</span>
          <span className="text-slate-300 font-mono italic">{node.modelVersion}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-slate-500 font-medium">Framework</span>
          <span className="text-slate-300">{node.framework || 'ONNX Runtime'}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-slate-500 font-medium">Location</span>
          <span className="text-slate-300">{node.location}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 mt-5">
        <button
          onClick={async (e) => {
            e.stopPropagation();
            if (onRestart) {
              try {
                await onRestart(node.id);
                toast('success', 'Restart Triggered', `Restart signal sent to ${node.nodeCode}. Logged to Deployments.`);
              } catch (err: any) {
                toast('error', 'Restart Failed', err.message);
              }
            }
          }}
          className="flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-700 hover:text-emerald-400 text-slate-300 py-2.5 rounded-lg text-sm font-bold transition-all shadow-inner"
        >
          <RefreshCw size={14} /> Restart
        </button>
        <button
          onClick={(e) => { 
            e.stopPropagation(); 
            // Log streaming feature - would require WebSocket connection to edge node
            // For now, show informative message about where logs can be accessed
            const logMessage = `Log Streaming for ${node.nodeCode}:\n\n` +
              `• SSH Access: ssh pi@${node.ipAddress || 'node-ip'}\n` +
              `• Log Location: /var/log/iot-edge/\n` +
              `• Real-time: tail -f /var/log/iot-edge/inference.log\n\n` +
              `WebSocket-based log streaming will be available in a future update.`;
            alert(logMessage);
          }}
          className="flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-700 hover:text-blue-400 text-slate-300 py-2.5 rounded-lg text-sm font-bold transition-all shadow-inner"
        >
          <FileText size={14} /> Logs
        </button>
      </div>
    </motion.div>
  );
});
export const NodeCardSkeleton: React.FC = () => (
  <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 animate-pulse space-y-4">
    <div className="flex items-center gap-3">
      <div className="w-2.5 h-2.5 rounded-full bg-slate-700" />
      <div className="h-4 bg-slate-800 rounded w-24" />
    </div>
    <div className="space-y-2">
      {[...Array(4)].map((_, i) => <div key={i} className="h-10 bg-slate-800/60 rounded" />)}
    </div>
    <div className="space-y-2 pt-3 border-t border-slate-800">
      {[...Array(4)].map((_, i) => <div key={i} className="h-3 bg-slate-800 rounded w-full" />)}
    </div>
  </div>
);
