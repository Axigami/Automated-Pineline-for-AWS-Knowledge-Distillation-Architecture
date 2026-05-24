import { useState } from 'react';
import type { ModelVersionRow, RetrainJobRow } from './types';

export function useMlopsStore() {
  const [metrics, setMetrics] = useState<ModelVersionRow[]>([]);
  const [activeJob, setActiveJob] = useState<RetrainJobRow | null>(null);
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  return {
    metrics, setMetrics,
    activeJob, setActiveJob,
    activeTaskId, setActiveTaskId,
    isLoading, setIsLoading,
    error, setError,
  };
}
