import { useCallback, useEffect, useRef, useState } from 'react';
import { supabase } from '../lib/supabaseClient';
import { useAuthContext } from '../auth/AuthProvider';

export interface AppNotification {
  id: string;
  title: string;
  message: string;
  severity: string;
  createdAt: string;
  isRead: boolean;
}

const EPOCH_ISO = new Date(0).toISOString();

function storageKeys(userId: string) {
  return {
    lastRead: `synapse_iot_last_read_ts:${userId}`,
    readIds: `synapse_iot_read_ids:${userId}`,
    unreadIds: `synapse_iot_unread_ids:${userId}`,
  } as const;
}

/** Logic đọc/chưa đọc — dùng chung cho fetch và realtime. */
export function checkIsReadPure(
  createdAt: string,
  id: string,
  lastReadTs: string,
  readIds: Set<string>,
  unreadIds: Set<string>,
): boolean {
  if (unreadIds.has(id)) return false;
  return (
    new Date(createdAt).getTime() <= new Date(lastReadTs).getTime() || readIds.has(id)
  );
}

export function useNotifications() {
  const { user } = useAuthContext();
  const userId = user?.id ?? null;

  const [notifications, setNotifications] = useState<AppNotification[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [lastReadTs, setLastReadTs] = useState<string>(EPOCH_ISO);
  const [readIds, setReadIds] = useState<Set<string>>(() => new Set());
  const [unreadIds, setUnreadIds] = useState<Set<string>>(() => new Set());

  const lastReadTsRef = useRef(lastReadTs);
  const readIdsRef = useRef(readIds);
  const unreadIdsRef = useRef(unreadIds);
  lastReadTsRef.current = lastReadTs;
  readIdsRef.current = readIds;
  unreadIdsRef.current = unreadIds;

  const checkIsRead = useCallback(
    (createdAt: string, id: string) =>
      checkIsReadPure(createdAt, id, lastReadTs, readIds, unreadIds),
    [lastReadTs, readIds, unreadIds],
  );

  useEffect(() => {
    if (!userId) {
      setNotifications([]);
      setUnreadCount(0);
      setLastReadTs(EPOCH_ISO);
      setReadIds(new Set());
      setUnreadIds(new Set());
      return;
    }

    const k = storageKeys(userId);
    const lr = localStorage.getItem(k.lastRead) || EPOCH_ISO;
    const r = new Set(JSON.parse(localStorage.getItem(k.readIds) || '[]') as string[]);
    const u = new Set(JSON.parse(localStorage.getItem(k.unreadIds) || '[]') as string[]);
    setLastReadTs(lr);
    setReadIds(r);
    setUnreadIds(u);

    let cancelled = false;

    const runFetch = async () => {
      try {
        const { data, error } = await supabase
          .from('alerts_all')
          .select('alert_id, alert_threat_type, alert_source_ip, alert_severity, alert_created_at')
          .order('alert_created_at', { ascending: false })
          .limit(30);

        if (cancelled) return;
        if (error) {
          console.error('Error fetching notifications:', error);
          return;
        }

        if (data) {
          const mapped: AppNotification[] = (data as any[]).map((row) => ({
            id: row.alert_id,
            title: `New ${row.alert_threat_type} detected`,
            message: `Suspicious activity from ${row.alert_source_ip} (Severity: ${row.alert_severity})`,
            severity: row.alert_severity,
            createdAt: row.alert_created_at,
            isRead: checkIsReadPure(row.alert_created_at, row.alert_id, lr, r, u),
          }));

          setNotifications(mapped);
          setUnreadCount(mapped.filter((n) => !n.isRead).length);
        }
      } catch (err) {
        console.error('Notification fetch failed', err);
      }
    };

    void runFetch();

    const channel = supabase
      .channel(`notifications-${userId}`)
      .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'alerts_all' }, (payload) => {
        const newRow = payload.new as Record<string, unknown>;
        const newNotif: AppNotification = {
          id: String(newRow.alert_id),
          title: `New ${newRow.alert_threat_type} detected`,
          message: `Suspicious activity from ${newRow.alert_source_ip} (Severity: ${newRow.alert_severity})`,
          severity: String(newRow.alert_severity),
          createdAt: String(newRow.alert_created_at),
          isRead: false,
        };

        setNotifications((prev) => {
          const updated = [newNotif, ...prev].slice(0, 30);
          setUnreadCount(
            updated.filter(
              (n) =>
                !checkIsReadPure(
                  n.createdAt,
                  n.id,
                  lastReadTsRef.current,
                  readIdsRef.current,
                  unreadIdsRef.current,
                ),
            ).length,
          );
          return updated;
        });
      })
      .subscribe();

    return () => {
      cancelled = true;
      supabase.removeChannel(channel);
    };
  }, [userId]);

  const markAllAsRead = useCallback(() => {
    if (!userId) return;
    const now = new Date().toISOString();
    const k = storageKeys(userId);
    localStorage.setItem(k.lastRead, now);
    localStorage.setItem(k.readIds, '[]');
    localStorage.setItem(k.unreadIds, '[]');

    setLastReadTs(now);
    setReadIds(new Set());
    setUnreadIds(new Set());
    setNotifications((prev) => prev.map((n) => ({ ...n, isRead: true })));
    setUnreadCount(0);
  }, [userId]);

  const markAsRead = useCallback((id: string) => {
    if (!userId) return;
    const k = storageKeys(userId);

    setReadIds((prev) => {
      const nextRead = new Set(prev);
      nextRead.add(id);
      setUnreadIds((prevU) => {
        const nextUnread = new Set(prevU);
        nextUnread.delete(id);
        localStorage.setItem(k.readIds, JSON.stringify(Array.from(nextRead)));
        localStorage.setItem(k.unreadIds, JSON.stringify(Array.from(nextUnread)));
        return nextUnread;
      });
      return nextRead;
    });

    setNotifications((prev) => {
      const updated = prev.map((n) => (n.id === id ? { ...n, isRead: true } : n));
      setUnreadCount(updated.filter((n) => !n.isRead).length);
      return updated;
    });
  }, [userId]);

  const markAsUnread = useCallback((id: string) => {
    if (!userId) return;
    const k = storageKeys(userId);

    setUnreadIds((prev) => {
      const nextUnread = new Set(prev);
      nextUnread.add(id);
      setReadIds((prevR) => {
        const nextRead = new Set(prevR);
        nextRead.delete(id);
        localStorage.setItem(k.unreadIds, JSON.stringify(Array.from(nextUnread)));
        localStorage.setItem(k.readIds, JSON.stringify(Array.from(nextRead)));
        return nextRead;
      });
      return nextUnread;
    });

    setNotifications((prev) => {
      const updated = prev.map((n) => (n.id === id ? { ...n, isRead: false } : n));
      setUnreadCount(updated.filter((n) => !n.isRead).length);
      return updated;
    });
  }, [userId]);

  return {
    notifications,
    unreadCount,
    markAllAsRead,
    markAsRead,
    markAsUnread,
    checkIsRead,
  };
}
