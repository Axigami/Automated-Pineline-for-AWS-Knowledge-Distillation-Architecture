import { AlertUIModel } from './types';
import { AggregatedAlert, AggregationConfig, EnhancedAlertUIModel } from './enhanced-types';

/**
 * Default aggregation configuration
 */
export const DEFAULT_AGGREGATION_CONFIG: AggregationConfig = {
  timeWindowMs: 300000, // 5 minutes
  groupByFields: ['sourceIp', 'attackType'],
};

/**
 * Aggregate alerts by source IP and attack type within a time window
 * 
 * Algorithm:
 * 1. Sort alerts by timestamp (newest first)
 * 2. Group alerts by sourceIp + attackType within time window
 * 3. Convert groups to AggregatedAlert objects
 * 
 * @param alerts - Array of alerts to aggregate
 * @param config - Aggregation configuration
 * @returns Array of aggregated alerts
 */
export function aggregateAlerts(
  alerts: AlertUIModel[],
  config: AggregationConfig = DEFAULT_AGGREGATION_CONFIG
): AggregatedAlert[] {
  // Handle empty input
  if (alerts.length === 0) {
    return [];
  }

  // 1. Sort alerts by timestamp (newest first)
  const sorted = [...alerts].sort((a, b) => {
    // Note: Dashboard AlertUIModel.time is HH:MM:SS, but it might be full ISO in some contexts
    // Or we should use a more reliable way to sort if they are all from the same day
    const timeA = new Date(`1970-01-01T${a.time}`).getTime();
    const timeB = new Date(`1970-01-01T${b.time}`).getTime();
    return timeB - timeA;
  });

  // 2. Group alerts by sourceIp + attackType within time window
  const groups = new Map<string, AlertUIModel[]>();
  const groupKeys = new Map<string, number>(); // Track group counters for same key

  for (const alert of sorted) {
    const isDoSOrDDoS = alert.label.toLowerCase().includes('dos');
    // For DoS/DDoS, group only by source IP, regardless of whether it's 'dos' or 'ddos'
    const baseKey = isDoSOrDDoS 
      ? `${alert.srcIp}_DOS_GROUP` 
      : `${alert.id}`;
    
    if (!isDoSOrDDoS) {
      groups.set(alert.id, [alert]);
      continue;
    }

    // Find existing group within time window
    let foundGroup = false;
    
    for (const [groupKey, group] of groups.entries()) {
      if (!groupKey.startsWith(baseKey)) continue;
      
      const latestInGroup = group[0];
      const timeDiff = Math.abs(
        new Date(`1970-01-01T${latestInGroup.time}`).getTime() - new Date(`1970-01-01T${alert.time}`).getTime()
      );
      
      if (timeDiff <= config.timeWindowMs) {
        group.push(alert);
        foundGroup = true;
        break;
      }
    }
    
    // Create new group if no matching group found
    if (!foundGroup) {
      const counter = groupKeys.get(baseKey) || 0;
      const uniqueKey = counter === 0 ? baseKey : `${baseKey}_${counter}`;
      groups.set(uniqueKey, [alert]);
      groupKeys.set(baseKey, counter + 1);
    }
  }

  // 3. Convert groups to AggregatedAlert objects
  const aggregated: AggregatedAlert[] = [];
  
  for (const group of groups.values()) {
    if (group.length === 1) {
      // Single alert - not aggregated
      aggregated.push(convertToAggregated(group[0], false));
    } else {
      // Multiple alerts - create aggregated group
      aggregated.push(createAggregatedGroup(group));
    }
  }

  return aggregated;
}

/**
 * Convert a single alert to AggregatedAlert format
 * 
 * @param alert - Single alert
 * @param isGroup - Whether this is part of a group
 * @returns AggregatedAlert object
 */
export function convertToAggregated(
  alert: AlertUIModel,
  isGroup: boolean
): AggregatedAlert {
  return {
    id: alert.id,
    isGroup,
    count: 1,
    alerts: [alert],
    sourceIp: alert.srcIp,
    targetIp: alert.targetIp || alert.srcIp || 'Unknown',
    attackType: alert.label,
    severity: alert.severity,
    firstSeen: alert.time,
    lastSeen: alert.time,
    status: alert.status,
  };
}

/**
 * Create an aggregated group from multiple alerts
 * 
 * @param alerts - Array of alerts to aggregate (must have at least 2)
 * @returns AggregatedAlert object
 */
export function createAggregatedGroup(alerts: AlertUIModel[]): AggregatedAlert {
  if (alerts.length < 2) {
    throw new Error('createAggregatedGroup requires at least 2 alerts');
  }

  // Sort by timestamp to get first and last
  const sortedByTime = [...alerts].sort((a, b) => {
    return new Date(`1970-01-01T${a.time}`).getTime() - new Date(`1970-01-01T${b.time}`).getTime();
  });

  const firstAlert = sortedByTime[0];
  const lastAlert = sortedByTime[sortedByTime.length - 1];

  // Determine highest severity in group
  const severity = getHighestSeverity(alerts.map(a => a.severity));

  // Generate group ID from first alert ID and count
  const groupId = `group_${firstAlert.id}_${alerts.length}`;

  return {
    id: groupId,
    isGroup: true,
    count: alerts.length,
    alerts,
    sourceIp: firstAlert.srcIp,
    targetIp: firstAlert.targetIp || firstAlert.srcIp || 'Unknown',
    attackType: firstAlert.label,
    severity,
    firstSeen: firstAlert.time,
    lastSeen: lastAlert.time,
    status: firstAlert.status,
  };
}

/**
 * Determine the highest severity from an array of severities
 * Priority: high > medium > low
 * 
 * @param severities - Array of severity levels
 * @returns Highest severity
 */
function getHighestSeverity(
  severities: ('high' | 'medium' | 'low')[]
): 'high' | 'medium' | 'low' {
  if (severities.includes('high')) return 'high';
  if (severities.includes('medium')) return 'medium';
  return 'low';
}

/**
 * Convert AlertUIModel to EnhancedAlertUIModel with aggregation fields
 * 
 * @param alert - Original alert
 * @param aggregated - Aggregated alert data
 * @returns Enhanced alert model
 */
export function enhanceAlert(
  alert: AlertUIModel,
  aggregated: AggregatedAlert
): EnhancedAlertUIModel {
  return {
    ...alert,
    isAggregated: aggregated.isGroup,
    aggregatedCount: aggregated.count,
    firstSeenAt: aggregated.firstSeen,
    lastSeenAt: aggregated.lastSeen,
    aggregatedAlerts: aggregated.isGroup ? aggregated.alerts : undefined,
    targetIp: aggregated.targetIp,
  };
}

/**
 * Sort aggregated alerts by severity (high first) and timestamp (newest first)
 * 
 * @param alerts - Array of aggregated alerts
 * @returns Sorted array
 */
export function sortAggregatedAlerts(alerts: AggregatedAlert[]): AggregatedAlert[] {
  const severityOrder = { high: 0, medium: 1, low: 2 };
  
  return [...alerts].sort((a, b) => {
    // First sort by severity
    const severityDiff = severityOrder[a.severity] - severityOrder[b.severity];
    if (severityDiff !== 0) return severityDiff;
    
    // Then sort by timestamp (newest first)
    const timeA = new Date(`1970-01-01T${a.lastSeen}`).getTime();
    const timeB = new Date(`1970-01-01T${b.lastSeen}`).getTime();
    return timeB - timeA;
  });
}

/**
 * Filter out dismissed alerts
 * 
 * @param alerts - Array of aggregated alerts
 * @returns Filtered array without dismissed alerts
 */
export function filterDismissedAlerts(alerts: AggregatedAlert[]): AggregatedAlert[] {
  return alerts.filter(alert => alert.status !== 'dismissed');
}
