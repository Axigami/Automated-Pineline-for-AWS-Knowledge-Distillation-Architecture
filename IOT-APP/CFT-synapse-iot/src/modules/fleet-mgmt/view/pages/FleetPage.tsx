import React from 'react';
import { useFleet } from '../../controller';
import { EdgeFleetManagement } from '../components/EdgeFleetManagement';

/**
 * FleetPage (Container Component)
 * - Gọi useFleet() để lấy dữ liệu thật từ Supabase
 * - Truyền data + actions xuống EdgeFleetManagement qua props
 */
const FleetPage: React.FC = () => {
  const {
    nodes,
    isLoading,
    error,
    telemetryMap,
    selectedNodeId,
    setSelectedNodeId,
    refresh,
    restartNode,
    addNode,
    deleteNode,
    updateNodeLocation,
  } = useFleet();

  return (
    <EdgeFleetManagement
      nodes={nodes}
      isLoading={isLoading}
      error={error}
      telemetryMap={telemetryMap}
      selectedNodeId={selectedNodeId}
      onSelectNode={setSelectedNodeId}
      onRefresh={refresh}
      onRestart={restartNode}
      onAddNode={addNode}
      onDelete={deleteNode}
      onEditLocation={updateNodeLocation}
    />
  );
};

export default FleetPage;
