import { create } from 'zustand';
import type {
  FlowUIModel,
  LabelAggregation,
  TopAttacker,
  TimelinePoint,
  FlowQueryParams,
} from './types';

const MAX_FLOW_BUFFER = 2000;

interface ThreatAnalyticsState {
  flows: FlowUIModel[];
  aggregation: LabelAggregation[];
  topAttackers: TopAttacker[];
  timeline: TimelinePoint[];
  queryParams: FlowQueryParams;
  isLoading: boolean;
  error: string | null;
  feedbackSuccess: string | null;

  setFlows: (flows: FlowUIModel[]) => void;
  setAggregation: (agg: LabelAggregation[]) => void;
  setTopAttackers: (attackers: TopAttacker[]) => void;
  setTimeline: (tl: TimelinePoint[]) => void;
  setQueryParams: (params: FlowQueryParams) => void;
  setIsLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setFeedbackSuccess: (msg: string | null) => void;
  updateFlowLabel: (flowId: string, trueLabel: string) => void;
  prependFlow: (flow: FlowUIModel) => void;
}

export const useThreatAnalyticsStore = create<ThreatAnalyticsState>((set) => ({
  flows: [],
  aggregation: [],
  topAttackers: [],
  timeline: [],
  queryParams: {
    query: '',
    from: new Date(Date.now() - 90 * 86400_000).toISOString(),
    to: new Date().toISOString(),
  },
  isLoading: false,
  error: null,
  feedbackSuccess: null,

  setFlows: (flows) => set({ flows }),
  setAggregation: (aggregation) => set({ aggregation }),
  setTopAttackers: (topAttackers) => set({ topAttackers }),
  setTimeline: (timeline) => set({ timeline }),
  setQueryParams: (queryParams) => set({ queryParams }),
  setIsLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
  setFeedbackSuccess: (feedbackSuccess) => set({ feedbackSuccess }),

  /** Cập nhật tức thì nhãn của 1 flow sau khi feedback (optimistic update) */
  updateFlowLabel: (flowId, trueLabel) =>
    set((state) => ({
      flows: state.flows.map((f) =>
        f.id === flowId ? { ...f, trueLabel, hasFeedback: true } : f
      ),
    })),

  /** Prepend 1 flow mới (từ realtime), giữ tối đa MAX_FLOW_BUFFER rows */
  prependFlow: (flow) =>
    set((state) => ({
      flows: [flow, ...state.flows].slice(0, MAX_FLOW_BUFFER),
    })),
}));
