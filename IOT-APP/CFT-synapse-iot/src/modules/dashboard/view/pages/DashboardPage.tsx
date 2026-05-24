import React from 'react';
import { useDashboard } from '../../controller';
import { DashboardOverview } from '../components/DashboardOverview';
import { useAuthContext } from '../../../../core/auth/AuthProvider';

/**
 * DashboardPage – Container component.
 * Lấy data từ Controller (useDashboard) và truyền xuống Presentational component.
 */
const DashboardPage: React.FC = () => {
  const {
    summary,
    recentAlerts,
    trafficSeries,
    homes,
    edgeNodes,
    attackDist,
    modelStats,
    networkFlowMetrics,
    isLoading,
    error,
    refresh,
    enableNode,
    dismissAlert,
  } = useDashboard();

  const { role, user } = useAuthContext();
  const isAdmin = role === 'admin' || user?.id === 'dced8a9f-5f89-4d44-a146-9f7070793749';
  
  // Debug: Log role
  console.log('[DashboardPage] Current role:', role);
  console.log('[DashboardPage] isAdmin:', isAdmin);

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-2">
        <p className="text-rose-400">{error}</p>
        <button onClick={refresh} className="text-blue-400 text-sm hover:underline">Thử lại</button>
      </div>
    );
  }

  return (
    <DashboardOverview
      summary={summary}
      recentAlerts={recentAlerts}
      trafficSeries={trafficSeries}
      homes={homes}
      edgeNodes={edgeNodes}
      attackDist={attackDist}
      modelStats={modelStats}
      networkFlowMetrics={networkFlowMetrics}
      isLoading={isLoading}
      isAdmin={isAdmin}
      onRefresh={refresh}
      enableNode={enableNode}
      dismissAlert={dismissAlert}
    />
  );
};

export default DashboardPage;
