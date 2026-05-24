import React from 'react';
import { Map as MapIcon } from 'lucide-react';
import type { EdgeNodeUIModel } from '../../model/types';

// Simple deterministic position from node code hash
function hashPosition(str: string): { x: number; y: number } {
  let h = 0;
  for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) & 0x7fffffff;
  const x = 10 + ((h & 0xff) / 255) * 80; // 10%–90%
  const y = 10 + (((h >> 8) & 0xff) / 255) * 80;
  return { x, y };
}

const STATUS_DOT_STYLE: Record<'online' | 'offline', string> = {
  online: 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)] animate-pulse',
  offline: 'bg-rose-500 shadow-[0_0_6px_rgba(244,63,94,0.5)]',
};

interface FleetMapPanelProps {
  nodes: EdgeNodeUIModel[];
  selectedNodeId: string | null;
  onSelect: (id: string) => void;
}

export const FleetMapPanel: React.FC<FleetMapPanelProps> = ({ nodes, selectedNodeId, onSelect }) => {
  return (
    <div className="bg-slate-900/80 backdrop-blur-sm border border-slate-800 rounded-xl p-5 shadow-xl">
      <div className="flex items-center gap-2 mb-4">
        <MapIcon size={16} className="text-blue-400" />
        <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">Fleet Map</h3>
        <span className="ml-auto text-[10px] text-slate-500">{nodes.length} nodes</span>
      </div>

      <div className="aspect-square bg-slate-950 rounded-lg border border-slate-800 relative overflow-hidden">
        {/* Grid overlay */}
        <div className="absolute inset-0 opacity-15 bg-[radial-gradient(#1e293b_1px,transparent_1px)] [background-size:14px_14px]" />
        {/* Glow effect */}
        <div className="absolute inset-0 bg-gradient-to-b from-blue-900/5 to-transparent pointer-events-none" />

        {/* Dynamic node dots */}
        {nodes.map(node => {
          const { x, y } = hashPosition(node.nodeCode);
          const dotStyle = STATUS_DOT_STYLE[node.status === 'offline' ? 'offline' : 'online'];
          const isSelected = node.id === selectedNodeId;

          return (
            <button
              key={node.id}
              title={`${node.nodeCode} – ${node.location}`}
              onClick={() => onSelect(node.id)}
              className={`absolute w-2.5 h-2.5 rounded-full -translate-x-1/2 -translate-y-1/2 transition-all duration-200 ${dotStyle}
                ${isSelected ? 'scale-150 ring-2 ring-white/40' : 'hover:scale-125'}
              `}
              style={{ left: `${x}%`, top: `${y}%` }}
            />
          );
        })}

        {/* Legend */}
        <div className="absolute bottom-2 left-2 right-2 flex items-center gap-3 bg-slate-900/80 backdrop-blur-sm border border-slate-800/80 rounded px-2 py-1">
          <span className="text-[9px] text-slate-500 font-bold uppercase tracking-widest mr-1">Legend</span>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" />
            <span className="text-[9px] text-slate-500">Online</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-rose-500 inline-block" />
            <span className="text-[9px] text-slate-500">Offline</span>
          </div>
        </div>
      </div>

      {/* Region label */}
      <div className="mt-2 text-center text-[10px] text-slate-600 font-mono">
        Region: Vietnam / Southeast Asia
      </div>
    </div>
  );
};
