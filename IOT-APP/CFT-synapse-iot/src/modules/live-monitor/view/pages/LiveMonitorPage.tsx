import React from 'react';
import { LiveMonitor } from '../components/LiveMonitor';
import { useLiveMonitor } from '../../controller';

/**
 * LiveMonitorPage – Container component.
 * Khởi tạo controller và truyền dữ liệu realtime xuống LiveMonitor (View).
 *
 * Data flow:
 *   Supabase Realtime → useLiveMonitor() → LiveMonitorPage → LiveMonitor (props)
 */
const LiveMonitorPage: React.FC = () => {
  const controller = useLiveMonitor();

  return (
    <LiveMonitor
      flows={controller.flows}
      alerts={controller.alerts}
      syncStatuses={controller.syncStatuses}
      radarData={controller.radarData}
      isPaused={controller.isPaused}
      onTogglePause={controller.togglePause}
      onVerifyAlert={controller.verifyAlert}
      error={controller.error}
      totalFlowsReceived={controller.totalFlowsReceived}
      verifyBanner={controller.verifyBanner}
      onDismissVerifyBanner={controller.dismissVerifyBanner}
    />
  );
};

export default LiveMonitorPage;
