import React, { useState } from 'react';
import { useAuthContext } from '../../../../core/auth/AuthProvider';
import { UserRolesTable } from '../../../settings/view/components/UserRolesTable';
import { useSettings } from '../../../settings/controller';
import { ChangePasswordModal } from '../components/ChangePasswordModal';
import { User, Mail, ShieldAlert, Calendar, Fingerprint, Lock } from 'lucide-react';
import { useLanguage } from '../../../../core/i18n/LanguageContext';

export const ProfilePage = () => {
  const { t } = useLanguage();
  const { user, role } = useAuthContext();
  const { userSettings, isLoading } = useSettings();
  const [passwordModalOpen, setPasswordModalOpen] = useState(false);

  const firstLetter = user?.email?.charAt(0).toUpperCase() || 'A';
  const email = user?.email || 'Unknown Email';
  
  // Restricted user configuration
  const RESTRICTED_USER_ID = 'f5333890-9353-4db9-9f41-6ad692480e50';
  const isRestrictedUser = user?.id === RESTRICTED_USER_ID;
  const displayRole = isRestrictedUser ? 'User' : (role || 'System Administrator');
  
  // Format creation date if available
  const joinedDate = user?.created_at 
    ? new Date(user.created_at).toLocaleDateString() 
    : 'N/A';

  return (
    <div className="max-w-7xl mx-auto flex flex-col h-full gap-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-slate-100 flex items-center gap-3">
          <User className="text-blue-500" size={32} />
          {t('profile', 'title') || 'My Profile & Access'}
        </h1>
        <p className="text-slate-400 mt-2">{t('profile', 'subTitle') || 'Manage your personal information, security credentials, and system roles.'}</p>
      </div>

      {/* Personal Information Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl shadow-xl overflow-hidden relative">
        <div className="h-32 bg-gradient-to-r from-blue-900/60 to-indigo-900/40 relative">
          <div className="absolute inset-0 bg-grid-slate-800/[0.04] bg-[size:20px_20px]"></div>
        </div>
        
        <div className="px-8 pb-8">
          <div className="flex flex-col sm:flex-row items-center sm:items-end gap-6 -mt-12 mb-6">
            <div className="w-24 h-24 rounded-2xl bg-slate-800 border-4 border-slate-900 shadow-2xl flex items-center justify-center flex-shrink-0 relative">
              <span className="text-blue-400 font-black text-4xl uppercase select-none">{firstLetter}</span>
              <div className="absolute -bottom-2 -right-2 bg-emerald-500/20 border-2 border-slate-900 px-2.5 py-0.5 rounded-full flex items-center gap-1.5 shadow-lg">
                <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.8)]"></span>
                <span className="text-[9px] font-bold uppercase tracking-widest text-emerald-400">Online</span>
              </div>
            </div>
            
            <div className="flex-1 text-center sm:text-left">
              <h2 className="text-2xl font-bold text-slate-100">{email.split('@')[0]}</h2>
              <div className="flex items-center justify-center sm:justify-start gap-4 mt-2">
                <span className="flex items-center gap-1.5 text-sm text-slate-400">
                  <Mail size={14} /> {email}
                </span>
                <span className="flex items-center gap-1.5 text-sm text-blue-400 font-medium">
                  <ShieldAlert size={14} /> {displayRole}
                </span>
              </div>
            </div>
            
            <div className="flex items-center gap-3 mt-4 sm:mt-0">
              <button
                type="button"
                onClick={() => setPasswordModalOpen(true)}
                className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 px-4 py-2 rounded-lg text-sm font-semibold transition-colors"
              >
                <Lock size={16} className="text-slate-400" /> {t('profile', 'changePassword') || 'Change Password'}
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-8 pt-8 border-t border-slate-800/60">
            <div className="bg-slate-950/50 rounded-lg p-4 border border-slate-800/40">
              <div className="flex items-center gap-2 text-slate-500 mb-2">
                <Fingerprint size={16} />
                <span className="text-xs font-bold uppercase tracking-wider">{t('profile', 'userIdLabel') || 'User ID'}</span>
              </div>
              <p className="font-mono text-sm text-slate-300 truncate" title={user?.id}>{user?.id || '—'}</p>
            </div>
            <div className="bg-slate-950/50 rounded-lg p-4 border border-slate-800/40">
              <div className="flex items-center gap-2 text-slate-500 mb-2">
                <Calendar size={16} />
                <span className="text-xs font-bold uppercase tracking-wider">{t('profile', 'joinedDate') || 'Joined Date'}</span>
              </div>
              <p className="font-medium text-sm text-slate-300">{joinedDate}</p>
            </div>
            <div className="bg-slate-950/50 rounded-lg p-4 border border-slate-800/40">
              <div className="flex items-center gap-2 text-slate-500 mb-2">
                <ShieldAlert size={16} />
                <span className="text-xs font-bold uppercase tracking-wider">{t('profile', 'roleLabel') || 'System Access'}</span>
              </div>
              <p className="font-medium text-sm text-slate-300 capitalize">{isRestrictedUser ? 'User' : (role || 'Administrator')}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Roles & Permissions Section */}
      <div className="mt-4">
        <h2 className="text-lg font-bold text-slate-200 mb-4 flex items-center gap-2">
          {t('profile', 'orgDirectory') || 'Organization Directory (Roles)'}
        </h2>
        {isLoading ? (
          <div className="p-8 text-center text-slate-500 text-sm bg-slate-900 border border-slate-800 rounded-xl">
             Loading directory data...
          </div>
        ) : (
          <UserRolesTable users={userSettings} />
        )}
      </div>

      {user?.email && (
        <ChangePasswordModal
          open={passwordModalOpen}
          onClose={() => setPasswordModalOpen(false)}
          userEmail={user.email}
        />
      )}
    </div>
  );
};
