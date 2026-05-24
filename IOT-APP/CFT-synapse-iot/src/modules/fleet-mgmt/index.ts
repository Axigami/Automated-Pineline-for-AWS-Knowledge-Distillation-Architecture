// Fleet Management Module - Public API

// View
export { EdgeFleetManagement } from './view';
export { default as FleetPage } from './view/pages/FleetPage';

// Model (Types)
export type {
  EdgeNodeRow,
  EdgeNodeUIModel,
  NodeTelemetryRow,
  TelemetryPoint,
  NodeStatus,
} from './model/types';

// Controller (Hook)
export { useFleet } from './controller';
