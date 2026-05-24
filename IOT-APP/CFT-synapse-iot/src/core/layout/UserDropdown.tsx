import React, { useState, useRef, useEffect } from 'react';
import { LogOut, User, Settings, ShieldAlert, ChevronDown } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useNavigate } from 'react-router-dom';
import { useAuthContext } from '../auth/AuthProvider';
import { supabase } from '../lib/supabaseClient';
import { UserAvatar } from '../components/UserAvatar';

export const UserDropdown = () => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const { user, role } = useAuthContext();

  const handleLogout = async () => {
    await supabase.auth.signOut();
    navigate('/login', { replace: true });
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const email = user?.email || 'Administrator';
  const username = email.split('@')[0];
  
  // Restricted user configuration
  const RESTRICTED_USER_ID = 'f5333890-9353-4db9-9f41-6ad692480e50';
  const isRestrictedUser = user?.id === RESTRICTED_USER_ID;
  const displayRole = isRestrictedUser ? 'User' : (role || 'System Administrator');

  return (
    <div className="relative" ref={dropdownRef}>
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-3 hover:bg-slate-800/80 pl-2 pr-3 py-1.5 rounded-full border transition-all group outline-none ${
          isOpen ? 'bg-slate-800/80 border-slate-700' : 'border-transparent hover:border-slate-700'
        }`}
      >
        <UserAvatar email={email} size="sm" className="group-hover:border-blue-400 transition-colors" />
        <div className="hidden md:flex flex-col items-start px-1 text-left">
          <span className="text-sm font-semibold text-slate-200 leading-tight mb-1 group-hover:text-white transition-colors">
            {username}
          </span>
          <span className="text-[10px] font-bold uppercase tracking-widest text-emerald-400 leading-none flex items-center gap-1.5 mt-0.5">
            <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.8)]"></span>
            Connected
          </span>
        </div>
        <ChevronDown size={14} className={`text-slate-500 group-hover:text-slate-300 ml-1 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 mt-2 w-64 bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden z-50 flex flex-col"
          >
            {/* Header Info */}
            <div className="px-4 py-4 border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm flex items-center gap-3">
               <UserAvatar email={email} size="md" />
               <div className="flex flex-col min-w-0">
                  <span className="text-sm font-bold text-slate-200 truncate">{email}</span>
                  <span className="text-xs text-slate-500 truncate capitalize flex items-center gap-1 mt-0.5">
                     <ShieldAlert size={12} className="text-blue-500" />
                     {displayRole}
                  </span>
               </div>
            </div>

            {/* Menu */}
            <div className="py-2 flex flex-col">
               <button 
                  onClick={() => { setIsOpen(false); navigate('/profile'); }}
                  className="w-full text-left px-4 py-2.5 text-sm text-slate-300 hover:bg-slate-800 hover:text-white transition-colors flex items-center gap-3"
               >
                  <User size={16} className="text-slate-400" /> My Profile & Access
               </button>
               
               {/* Hide System Settings for restricted user */}
               {!isRestrictedUser && (
                 <button 
                    onClick={() => { setIsOpen(false); navigate('/settings'); }}
                    className="w-full text-left px-4 py-2.5 text-sm text-slate-300 hover:bg-slate-800 hover:text-white transition-colors flex items-center gap-3"
                 >
                    <Settings size={16} className="text-slate-400" /> System Settings
                 </button>
               )}
               
               <div className="h-px bg-slate-800/50 w-full my-2"></div>

               <button 
                  onClick={(e) => { e.stopPropagation(); setIsOpen(false); handleLogout(); }}
                  className="w-full text-left px-4 py-2.5 text-sm text-rose-400 hover:bg-rose-500/10 transition-colors flex items-center gap-3 font-semibold"
               >
                  <LogOut size={16} className="text-rose-400" /> Sign Out
               </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
