import { useState } from 'react';
import type { FlowInferenceRow, ModelComparisonRow } from './types';

export function useModelInsightsStore() {
  const [inferences, setInferences] = useState<FlowInferenceRow[]>([]);
  const [modelVersions, setModelVersions] = useState<ModelComparisonRow[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  return { inferences, setInferences, modelVersions, setModelVersions, isLoading, setIsLoading, error, setError };
}
