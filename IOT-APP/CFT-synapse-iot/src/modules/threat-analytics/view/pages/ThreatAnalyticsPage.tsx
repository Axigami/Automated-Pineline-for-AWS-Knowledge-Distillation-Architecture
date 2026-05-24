import React from 'react';
import { useThreatAnalytics } from '../../controller';
import { ThreatAnalytics } from '../components/ThreatAnalytics';
import type { FlowQueryParams, LabelFeedbackRequest } from '../../model/types';

const ThreatAnalyticsPage: React.FC = () => {
  const {
    flows,
    aggregation,
    topAttackers,
    timeline,
    queryParams,
    isLoading,
    error,
    feedbackSuccess,
    setQueryParams,
    queryLogs,
    submitFeedback,
    availableHomes,
  } = useThreatAnalytics();

  return (
    <ThreatAnalytics
      flows={flows}
      aggregation={aggregation}
      topAttackers={topAttackers}
      timeline={timeline}
      queryParams={queryParams}
      isLoading={isLoading}
      error={error}
      feedbackSuccess={feedbackSuccess}
      availableHomes={availableHomes}
      onQueryParamsChange={(params: FlowQueryParams) => setQueryParams(params)}
      onSearch={queryLogs}
      onSubmitFeedback={(req: LabelFeedbackRequest) => submitFeedback(req)}
    />
  );
};

export default ThreatAnalyticsPage;
