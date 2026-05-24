import React, { useState, useRef, useEffect } from 'react';
import { Bell, Check, AlertTriangle, AlertCircle, Info, X } from 'lucide-react';
import { useNotifications, AppNotification } from '../hooks/useNotifications';
import { motion, AnimatePresence } from 'motion/react';
import { useNavigate } from 'react-router-dom';

const getSeverityIcon = (severity: string) => {
  if (severity === 'critical' || severity === 'high') {
    return <AlertTriangle size={16} className="text-rose-400" />;
  }
  if (severity === 'medium') {
    return <AlertCircle size={16} className="text-amber-400" />;
  }
  return <Info size={16} className="text-blue-400" />;
};

const getSeverityBg = (severity: string) => {
  if (severity === 'critical' || severity === 'high') return 'bg-rose-500/10 border-rose-500/20';
  if (severity === 'medium') return 'bg-amber-500/10 border-amber-500/20';
  return 'bg-blue-500/10 border-blue-500/20';
};

const formatTimeAgo = (dateStr: string) => {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
};

export const NotificationDropdown = () => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  
  const { notifications, unreadCount, markAllAsRead, markAsRead } = useNotifications();

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleNotificationClick = (notif: AppNotification) => {
    markAsRead(notif.id);
    setIsOpen(false);
    // Navigate to dashboard where alerts are displayed.
    navigate('/');
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className={`relative p-2 rounded-lg transition-all ${
          isOpen ? 'bg-slate-800 text-slate-200' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
        }`}
      >
        <Bell size={20} />
        {unreadCount > 0 && (
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-rose-500 rounded-full border border-slate-950 animate-pulse"></span>
        )}
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 mt-2 w-80 sm:w-96 bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden z-50 flex flex-col"
          >
            {/* Header */}
            <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between bg-slate-900/50 backdrop-blur-sm">
              <h3 className="font-bold text-slate-200 flex items-center gap-2">
                Notifications
                {unreadCount > 0 && (
                  <span className="bg-rose-500 text-white text-[10px] font-black px-1.5 py-0.5 rounded-full">{unreadCount}</span>
                )}
              </h3>
              <div className="flex gap-2">
                <button 
                  onClick={markAllAsRead} 
                  disabled={unreadCount === 0}
                  className="text-[10px] font-bold text-slate-400 hover:text-blue-400 uppercase tracking-widest transition-colors disabled:opacity-50 disabled:hover:text-slate-400"
                >
                  Mark All Read
                </button>
              </div>
            </div>

            {/* List */}
            <div className="max-h-[400px] overflow-y-auto custom-scrollbar">
              {notifications.length === 0 ? (
                <div className="p-8 text-center flex flex-col items-center gap-2 text-slate-500">
                  <Bell size={28} className="opacity-20 mb-2" />
                  <p className="text-sm font-semibold text-slate-400">All caught up!</p>
                  <p className="text-xs">No pending notifications.</p>
                </div>
              ) : (
                <div className="divide-y divide-slate-800/50">
                  {notifications.map((notif) => (
                    <div 
                      key={notif.id}
                      onClick={() => handleNotificationClick(notif)}
                      className={`p-4 hover:bg-slate-800/50 transition-colors cursor-pointer group relative ${!notif.isRead ? 'bg-slate-900/80' : 'opacity-70'}`}
                    >
                      {!notif.isRead && (
                        <div className="absolute left-0 top-0 bottom-0 w-1 bg-blue-500 rounded-r-full" />
                      )}
                      <div className="flex gap-3">
                        <div className={`mt-0.5 p-1.5 rounded-lg border flex-shrink-0 ${getSeverityBg(notif.severity)}`}>
                          {getSeverityIcon(notif.severity)}
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="flex justify-between items-start mb-1">
                            <h4 className={`text-sm font-semibold truncate ${!notif.isRead ? 'text-slate-200' : 'text-slate-300'}`}>
                              {notif.title}
                            </h4>
                            <span className="text-[10px] font-mono text-slate-500 whitespace-nowrap ml-2 mt-0.5">
                              {formatTimeAgo(notif.createdAt)}
                            </span>
                          </div>
                          <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">
                            {notif.message}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            <div className="p-2 border-t border-slate-800 text-center bg-slate-900">
              <button 
                onClick={() => { setIsOpen(false); navigate('/alerts'); }}
                className="text-xs font-bold text-slate-500 hover:text-blue-400 transition-colors py-1.5 w-full uppercase tracking-widest"
              >
                View alert history
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
