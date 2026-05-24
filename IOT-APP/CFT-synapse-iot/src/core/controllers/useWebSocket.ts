import { useEffect, useRef, useCallback } from 'react';

type MessageHandler<T> = (data: T) => void;

interface UseWebSocketOptions<T> {
  url: string;
  onMessage: MessageHandler<T>;
  enabled?: boolean;
  reconnectDelay?: number;
}

/**
 * useWebSocket
 * Global hook quản lý kết nối WebSocket với cơ chế tự động reconnect.
 * Dùng cho Live Monitor (FR 1.1) và Retraining Progress (FR 4.2).
 */
export function useWebSocket<T = unknown>({
  url,
  onMessage,
  enabled = true,
  reconnectDelay = 3000,
}: UseWebSocketOptions<T>) {
  const wsRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isMountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!enabled || !isMountedRef.current) return;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data) as T;
        onMessage(parsed);
      } catch {
        console.warn('[useWebSocket] Failed to parse message:', event.data);
      }
    };

    ws.onclose = () => {
      if (isMountedRef.current) {
        timerRef.current = setTimeout(connect, reconnectDelay);
      }
    };

    ws.onerror = (err) => {
      console.error('[useWebSocket] Error:', err);
      ws.close();
    };
  }, [url, enabled, onMessage, reconnectDelay]);

  useEffect(() => {
    isMountedRef.current = true;
    connect();

    return () => {
      isMountedRef.current = false;
      if (timerRef.current) clearTimeout(timerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { send };
}
