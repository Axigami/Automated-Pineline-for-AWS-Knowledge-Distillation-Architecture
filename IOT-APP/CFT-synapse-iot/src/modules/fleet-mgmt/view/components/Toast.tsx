import React, { useEffect, useState, useCallback } from 'react';
import { CheckCircle, XCircle, AlertTriangle, Info, X } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// ─── Toast Types ────────────────────────────────────────────────────────────────

export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface ToastMessage {
  id: string;
  type: ToastType;
  title: string;
  description?: string;
  duration?: number; // ms, default 4000
}

// ─── Global Toast State (lightweight pub/sub) ───────────────────────────────────

type Listener = (toasts: ToastMessage[]) => void;

let toasts: ToastMessage[] = [];
const listeners = new Set<Listener>();

function emit() {
  listeners.forEach((fn) => fn([...toasts]));
}

export function toast(type: ToastType, title: string, description?: string, duration = 4000) {
  const msg: ToastMessage = {
    id: crypto.randomUUID(),
    type,
    title,
    description,
    duration,
  };
  toasts = [...toasts, msg];
  emit();

  // Auto-remove after duration
  setTimeout(() => {
    toasts = toasts.filter((t) => t.id !== msg.id);
    emit();
  }, duration);
}

function removeToast(id: string) {
  toasts = toasts.filter((t) => t.id !== id);
  emit();
}

// ─── Toast Container Component ──────────────────────────────────────────────────

const ICON_MAP: Record<ToastType, React.ReactNode> = {
  success: <CheckCircle size={18} className="text-emerald-400" />,
  error: <XCircle size={18} className="text-red-400" />,
  warning: <AlertTriangle size={18} className="text-amber-400" />,
  info: <Info size={18} className="text-blue-400" />,
};

const BORDER_MAP: Record<ToastType, string> = {
  success: 'border-emerald-500/30',
  error: 'border-red-500/30',
  warning: 'border-amber-500/30',
  info: 'border-blue-500/30',
};

const GLOW_MAP: Record<ToastType, string> = {
  success: 'shadow-emerald-500/10',
  error: 'shadow-red-500/10',
  warning: 'shadow-amber-500/10',
  info: 'shadow-blue-500/10',
};

const BAR_MAP: Record<ToastType, string> = {
  success: 'bg-emerald-500',
  error: 'bg-red-500',
  warning: 'bg-amber-500',
  info: 'bg-blue-500',
};

export const ToastContainer: React.FC = () => {
  const [items, setItems] = useState<ToastMessage[]>([]);

  useEffect(() => {
    listeners.add(setItems);
    return () => { listeners.delete(setItems); };
  }, []);

  return (
    <div className="fixed top-5 right-5 z-[99999] flex flex-col gap-3 pointer-events-none max-w-sm w-full">
      <AnimatePresence mode="popLayout">
        {items.map((t) => (
          <motion.div
            key={t.id}
            layout
            initial={{ opacity: 0, x: 80, scale: 0.95 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 80, scale: 0.95 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className={`pointer-events-auto bg-slate-900/95 backdrop-blur-xl border ${BORDER_MAP[t.type]} rounded-xl shadow-2xl ${GLOW_MAP[t.type]} overflow-hidden`}
          >
            <div className="flex items-start gap-3 px-4 py-3">
              <div className="mt-0.5 shrink-0">{ICON_MAP[t.type]}</div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-bold text-slate-100 leading-tight">{t.title}</p>
                {t.description && (
                  <p className="text-xs text-slate-400 mt-1 leading-relaxed">{t.description}</p>
                )}
              </div>
              <button
                onClick={() => removeToast(t.id)}
                className="shrink-0 text-slate-500 hover:text-slate-200 transition-colors p-0.5"
              >
                <X size={14} />
              </button>
            </div>
            {/* Animated progress bar */}
            <div className="h-[2px] w-full bg-slate-800">
              <motion.div
                initial={{ width: '100%' }}
                animate={{ width: '0%' }}
                transition={{ duration: (t.duration ?? 4000) / 1000, ease: 'linear' }}
                className={`h-full ${BAR_MAP[t.type]}`}
              />
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
};
