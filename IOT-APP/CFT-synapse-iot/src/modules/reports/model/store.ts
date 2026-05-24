import { useState } from 'react';
import type { AuditLogEntry } from './types';

export function useReportsStore() {
  const [auditLogs, setAuditLogs] = useState<AuditLogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  return { auditLogs, setAuditLogs, isLoading, setIsLoading, error, setError };
}
