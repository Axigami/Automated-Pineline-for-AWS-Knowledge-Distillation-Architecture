import { create } from 'zustand';
import type { EdgeNodeUIModel, TelemetryPoint } from './types';

interface FleetState {
  nodes: EdgeNodeUIModel[];
  isLoading: boolean;
  error: string | null;
  selectedNodeId: string | null;
  telemetryMap: Record<string, TelemetryPoint[]>;

  setNodes: (nodes: EdgeNodeUIModel[]) => void;
  updateNode: (id: string, partial: Partial<EdgeNodeUIModel>) => void;
  setIsLoading: (isLoading: boolean) => void;
  setError: (error: string | null) => void;
  setSelectedNodeId: (id: string | null) => void;
  setTelemetryMap: (map: Record<string, TelemetryPoint[]>) => void;
  appendTelemetryPoint: (nodeId: string, point: TelemetryPoint) => void;
}

export const useFleetStore = create<FleetState>((set) => ({
  nodes: [],
  isLoading: false,
  error: null,
  selectedNodeId: null,
  telemetryMap: {},

  setNodes: (nodes) => set({ nodes }),
  updateNode: (id, partial) => set((state) => ({
    nodes: state.nodes.map((n) => (n.id === id ? { ...n, ...partial } : n)),
  })),
  setIsLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
  setSelectedNodeId: (id) => set({ selectedNodeId: id }),
  setTelemetryMap: (map) => set({ telemetryMap: map }),
  appendTelemetryPoint: (nodeId, point) => set((state) => {
    const existing = state.telemetryMap[nodeId] || [];
    const updated = [...existing, point].slice(-20);
    return {
      telemetryMap: { ...state.telemetryMap, [nodeId]: updated },
    };
  }),
}));
