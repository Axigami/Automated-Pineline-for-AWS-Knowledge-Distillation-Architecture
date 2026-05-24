/**
 * Property-Based Tests for Alert Aggregation
 * 
 * Feature: dashboard-ui-improvements
 * Property 1: Alert aggregation groups by source IP and attack type within time window
 * 
 * NOTE: Requires fast-check and vitest to be installed:
 *   npm install --save-dev fast-check vitest @vitest/ui
 */

import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import { aggregateAlerts, DEFAULT_AGGREGATION_CONFIG } from './alert-aggregation';
import { AlertUIModel } from './types';

/**
 * Arbitrary generator for AlertUIModel
 * Generates random alerts with varying properties for property-based testing
 */
function alertArbitrary(): fc.Arbitrary<AlertUIModel> {
  return fc.record({
    id: fc.uuid(),
    srcIp: fc.ipV4(),
    targetIp: fc.ipV4(),
    label: fc.constantFrom('DDoS', 'Port Scan', 'Malware', 'Brute Force', 'SQL Injection', 'XSS', 'MITM'),
    time: fc.integer({ min: 0, max: 86400000 }).map(ms => {
      const baseDate = new Date('2024-01-01T00:00:00Z').getTime();
      return new Date(baseDate + ms).toISOString();
    }),
    severity: fc.constantFrom('high', 'medium', 'low') as fc.Arbitrary<'high' | 'medium' | 'low'>,
    status: fc.constantFrom('pending', 'verified', 'dismissed'),
    confidencePct: fc.integer({ min: 50, max: 100 }).map(n => `${n}%`),
    confidenceVal: fc.double({ min: 0.5, max: 1.0 }),
    source: fc.constantFrom('raspi-nfstream', 'edge-node-01', 'edge-node-02'),
    seqValues: fc.array(fc.double({ min: 0, max: 1 }), { minLength: 5, maxLength: 10 }),
    seqSteps: fc.array(fc.string(), { minLength: 5, maxLength: 10 }),
    homeName: fc.constant('Home A'),
    homeCode: fc.constant('HM001'),
    deviceName: fc.constant('Device A'),
    locationName: fc.constant('Location A'),
    alert_node_id: fc.uuid(),
    alert_home_id: fc.uuid(),
  });
}

describe('Alert Aggregation - Property-Based Tests', () => {
  /**
   * Property 1: Alert aggregation groups by source IP and attack type within time window
   * 
   * Validates: Requirements 3.1
   * 
   * For any set of alerts, when aggregated with a time window of 5 minutes,
   * alerts with the same source IP address and attack type that occur within
   * the time window SHALL be grouped together, and alerts outside the time
   * window SHALL form separate groups.
   */
  it('Property 1: Aggregation groups by source IP and attack type within time window', () => {
    fc.assert(
      fc.property(
        fc.array(alertArbitrary(), { minLength: 1, maxLength: 100 }),
        (alerts) => {
          const aggregated = aggregateAlerts(alerts, DEFAULT_AGGREGATION_CONFIG);
          
          // Invariant 1 & 2: All alerts in a group have same source IP and attack type
          for (const group of aggregated) {
            const firstAlert = group.alerts[0];
            for (const alert of group.alerts) {
              expect(alert.srcIp).toBe(firstAlert.srcIp);
              expect(alert.label).toBe(firstAlert.label);
            }
          }
          
          // Invariant 3: All alerts in a group are within 5 minutes (300000ms)
          for (const group of aggregated) {
            const timestamps = group.alerts.map(a => new Date(a.time).getTime());
            const minTime = Math.min(...timestamps);
            const maxTime = Math.max(...timestamps);
            const timeDiff = maxTime - minTime;
            
            expect(timeDiff).toBeLessThanOrEqual(DEFAULT_AGGREGATION_CONFIG.timeWindowMs);
          }
          
          // Invariant 4: No two groups have the same (source IP, attack type) with overlapping time windows
          for (let i = 0; i < aggregated.length; i++) {
            for (let j = i + 1; j < aggregated.length; j++) {
              const groupA = aggregated[i];
              const groupB = aggregated[j];
              
              // If same source IP and attack type, time windows must not overlap
              if (groupA.sourceIp === groupB.sourceIp && groupA.attackType === groupB.attackType) {
                const timeA = new Date(groupA.lastSeen).getTime();
                const timeB = new Date(groupB.firstSeen).getTime();
                const timeDiff = Math.abs(timeA - timeB);
                
                // Time difference should be greater than window (groups are separate)
                expect(timeDiff).toBeGreaterThan(DEFAULT_AGGREGATION_CONFIG.timeWindowMs);
              }
            }
          }
          
          // Invariant 5: Total alert count preserved
          const totalAlertsInGroups = aggregated.reduce((sum, g) => sum + g.count, 0);
          expect(totalAlertsInGroups).toBe(alerts.length);
        }
      ),
      { 
        numRuns: 100, // Run 100 iterations
        verbose: true, // Show detailed output on failure
      }
    );
  });

  /**
   * Additional property: Aggregation is deterministic
   * 
   * Running aggregation twice on the same input should produce the same result
   */
  it('Property 2: Aggregation is deterministic', () => {
    fc.assert(
      fc.property(
        fc.array(alertArbitrary(), { minLength: 1, maxLength: 50 }),
        (alerts) => {
          const result1 = aggregateAlerts(alerts, DEFAULT_AGGREGATION_CONFIG);
          const result2 = aggregateAlerts(alerts, DEFAULT_AGGREGATION_CONFIG);
          
          // Same number of groups
          expect(result1.length).toBe(result2.length);
          
          // Same group sizes
          const sizes1 = result1.map(g => g.count).sort((a, b) => a - b);
          const sizes2 = result2.map(g => g.count).sort((a, b) => a - b);
          expect(sizes1).toEqual(sizes2);
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Additional property: Single alerts are not marked as groups
   * 
   * When an alert cannot be grouped with any other alert, it should have isGroup = false
   */
  it('Property 3: Single alerts are not marked as groups', () => {
    fc.assert(
      fc.property(
        fc.array(alertArbitrary(), { minLength: 1, maxLength: 50 }),
        (alerts) => {
          const aggregated = aggregateAlerts(alerts, DEFAULT_AGGREGATION_CONFIG);
          
          for (const group of aggregated) {
            if (group.count === 1) {
              expect(group.isGroup).toBe(false);
            } else {
              expect(group.isGroup).toBe(true);
            }
          }
        }
      ),
      { numRuns: 100 }
    );
  });
});

describe('Alert Aggregation - Edge Cases', () => {
  /**
   * Test: Empty input returns empty array
   */
  it('should return empty array for empty input', () => {
    const aggregated = aggregateAlerts([], DEFAULT_AGGREGATION_CONFIG);
    expect(aggregated).toEqual([]);
  });

  /**
   * Test: Single alert returns single non-grouped item
   */
  it('should return single group for single alert', () => {
    const alert: AlertUIModel = {
      id: '1',
      srcIp: '192.168.1.100',
      targetIp: '192.168.1.1',
      label: 'DDoS',
      time: '2024-01-01T10:00:00Z',
      severity: 'high',
      status: 'pending',
      confidencePct: '95%',
      confidenceVal: 0.95,
      source: 'raspi-nfstream',
      seqValues: [0.1, 0.2, 0.3],
      seqSteps: ['step1', 'step2', 'step3'],
      homeName: 'Home A',
      homeCode: 'HM001',
      deviceName: 'Device A',
      locationName: 'Location A',
      alert_node_id: 'node1',
      alert_home_id: 'home1',
    };
    
    const aggregated = aggregateAlerts([alert], DEFAULT_AGGREGATION_CONFIG);
    
    expect(aggregated).toHaveLength(1);
    expect(aggregated[0].count).toBe(1);
    expect(aggregated[0].isGroup).toBe(false);
    expect(aggregated[0].sourceIp).toBe('192.168.1.100');
    expect(aggregated[0].attackType).toBe('DDoS');
  });

  /**
   * Test: Alerts outside time window create separate groups
   */
  it('should create separate groups for alerts outside time window', () => {
    const alerts: AlertUIModel[] = [
      {
        id: '1',
        srcIp: '192.168.1.100',
        targetIp: '192.168.1.1',
        label: 'DDoS',
        time: '2024-01-01T10:00:00Z',
        severity: 'high',
        status: 'pending',
        confidencePct: '95%',
        confidenceVal: 0.95,
        source: 'raspi-nfstream',
        seqValues: [],
        seqSteps: [],
        homeName: 'Home A',
        homeCode: 'HM001',
        deviceName: 'Device A',
        locationName: 'Location A',
        alert_node_id: 'node1',
        alert_home_id: 'home1',
      },
      {
        id: '2',
        srcIp: '192.168.1.100',
        targetIp: '192.168.1.1',
        label: 'DDoS',
        time: '2024-01-01T10:10:00Z', // 10 minutes later (outside 5-minute window)
        severity: 'high',
        status: 'pending',
        confidencePct: '90%',
        confidenceVal: 0.90,
        source: 'raspi-nfstream',
        seqValues: [],
        seqSteps: [],
        homeName: 'Home A',
        homeCode: 'HM001',
        deviceName: 'Device A',
        locationName: 'Location A',
        alert_node_id: 'node1',
        alert_home_id: 'home1',
      },
    ];
    
    const aggregated = aggregateAlerts(alerts, DEFAULT_AGGREGATION_CONFIG);
    
    expect(aggregated).toHaveLength(2); // Two separate groups
    expect(aggregated[0].count).toBe(1);
    expect(aggregated[1].count).toBe(1);
  });

  /**
   * Test: Alerts with different source IPs create separate groups
   */
  it('should create separate groups for alerts with different source IPs', () => {
    const alerts: AlertUIModel[] = [
      {
        id: '1',
        srcIp: '192.168.1.100',
        targetIp: '192.168.1.1',
        label: 'DDoS',
        time: '2024-01-01T10:00:00Z',
        severity: 'high',
        status: 'pending',
        confidencePct: '95%',
        confidenceVal: 0.95,
        source: 'raspi-nfstream',
        seqValues: [],
        seqSteps: [],
        homeName: 'Home A',
        homeCode: 'HM001',
        deviceName: 'Device A',
        locationName: 'Location A',
        alert_node_id: 'node1',
        alert_home_id: 'home1',
      },
      {
        id: '2',
        srcIp: '192.168.1.200', // Different IP
        targetIp: '192.168.1.1',
        label: 'DDoS',
        time: '2024-01-01T10:01:00Z',
        severity: 'high',
        status: 'pending',
        confidencePct: '90%',
        confidenceVal: 0.90,
        source: 'raspi-nfstream',
        seqValues: [],
        seqSteps: [],
        homeName: 'Home A',
        homeCode: 'HM001',
        deviceName: 'Device A',
        locationName: 'Location A',
        alert_node_id: 'node1',
        alert_home_id: 'home1',
      },
    ];
    
    const aggregated = aggregateAlerts(alerts, DEFAULT_AGGREGATION_CONFIG);
    
    expect(aggregated).toHaveLength(2); // Two separate groups
  });

  /**
   * Test: Alerts with different attack types create separate groups
   */
  it('should create separate groups for alerts with different attack types', () => {
    const alerts: AlertUIModel[] = [
      {
        id: '1',
        srcIp: '192.168.1.100',
        targetIp: '192.168.1.1',
        label: 'DDoS',
        time: '2024-01-01T10:00:00Z',
        severity: 'high',
        status: 'pending',
        confidencePct: '95%',
        confidenceVal: 0.95,
        source: 'raspi-nfstream',
        seqValues: [],
        seqSteps: [],
        homeName: 'Home A',
        homeCode: 'HM001',
        deviceName: 'Device A',
        locationName: 'Location A',
        alert_node_id: 'node1',
        alert_home_id: 'home1',
      },
      {
        id: '2',
        srcIp: '192.168.1.100',
        targetIp: '192.168.1.1',
        label: 'Port Scan', // Different attack type
        time: '2024-01-01T10:01:00Z',
        severity: 'medium',
        status: 'pending',
        confidencePct: '85%',
        confidenceVal: 0.85,
        source: 'raspi-nfstream',
        seqValues: [],
        seqSteps: [],
        homeName: 'Home A',
        homeCode: 'HM001',
        deviceName: 'Device A',
        locationName: 'Location A',
        alert_node_id: 'node1',
        alert_home_id: 'home1',
      },
    ];
    
    const aggregated = aggregateAlerts(alerts, DEFAULT_AGGREGATION_CONFIG);
    
    expect(aggregated).toHaveLength(2); // Two separate groups
  });

  /**
   * Test: Alerts within time window are grouped together
   */
  it('should group alerts with same source IP and attack type within time window', () => {
    const alerts: AlertUIModel[] = [
      {
        id: '1',
        srcIp: '192.168.1.100',
        targetIp: '192.168.1.1',
        label: 'DDoS',
        time: '2024-01-01T10:00:00Z',
        severity: 'high',
        status: 'pending',
        confidencePct: '95%',
        confidenceVal: 0.95,
        source: 'raspi-nfstream',
        seqValues: [],
        seqSteps: [],
        homeName: 'Home A',
        homeCode: 'HM001',
        deviceName: 'Device A',
        locationName: 'Location A',
        alert_node_id: 'node1',
        alert_home_id: 'home1',
      },
      {
        id: '2',
        srcIp: '192.168.1.100',
        targetIp: '192.168.1.1',
        label: 'DDoS',
        time: '2024-01-01T10:02:00Z', // 2 minutes later (within 5-minute window)
        severity: 'high',
        status: 'pending',
        confidencePct: '90%',
        confidenceVal: 0.90,
        source: 'raspi-nfstream',
        seqValues: [],
        seqSteps: [],
        homeName: 'Home A',
        homeCode: 'HM001',
        deviceName: 'Device A',
        locationName: 'Location A',
        alert_node_id: 'node1',
        alert_home_id: 'home1',
      },
      {
        id: '3',
        srcIp: '192.168.1.100',
        targetIp: '192.168.1.1',
        label: 'DDoS',
        time: '2024-01-01T10:04:00Z', // 4 minutes after first (within window)
        severity: 'medium',
        status: 'pending',
        confidencePct: '88%',
        confidenceVal: 0.88,
        source: 'raspi-nfstream',
        seqValues: [],
        seqSteps: [],
        homeName: 'Home A',
        homeCode: 'HM001',
        deviceName: 'Device A',
        locationName: 'Location A',
        alert_node_id: 'node1',
        alert_home_id: 'home1',
      },
    ];
    
    const aggregated = aggregateAlerts(alerts, DEFAULT_AGGREGATION_CONFIG);
    
    expect(aggregated).toHaveLength(1); // Single aggregated group
    expect(aggregated[0].count).toBe(3);
    expect(aggregated[0].isGroup).toBe(true);
    expect(aggregated[0].sourceIp).toBe('192.168.1.100');
    expect(aggregated[0].attackType).toBe('DDoS');
  });
});
