import React, { useState } from 'react';
import { Users, ShieldCheck, Mail, Clock, ChevronDown, ChevronUp, UserPlus, MoreHorizontal, X, Check } from 'lucide-react';
import { useAuthContext } from '../../../../core/auth/AuthProvider';
import { supabase } from '../../../../core/lib/supabaseClient';
import type { UserRoleSetting } from '../../model/types';

interface UserRolesTableProps {
  users: UserRoleSetting[];
  onRefresh?: () => void;
}

/**
 * UserRolesTable – Presentational Component.
 * Hiển thị danh sách users, vai trò, settings từ bảng users_roles_settings dưới dạng Accordion.
 */
export const UserRolesTable: React.FC<UserRolesTableProps> = ({ users, onRefresh }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [showAddUserModal, setShowAddUserModal] = useState(false);
  const [newUserEmail, setNewUserEmail] = useState('');
  const [newUserPassword, setNewUserPassword] = useState('');
  const [newUserDisplayName, setNewUserDisplayName] = useState('');
  const [newUserRole, setNewUserRole] = useState<'admin' | 'operator' | 'viewer'>('viewer');
  const [isCreatingUser, setIsCreatingUser] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  
  const { role } = useAuthContext();
  const isAdmin = role === 'admin';

  const handleAddUser = async () => {
    if (!newUserEmail.trim() || !newUserPassword.trim()) {
      setCreateError('Email and password are required');
      return;
    }

    setIsCreatingUser(true);
    setCreateError(null);

    try {
      // 1. Create user in Supabase Auth
      const { data: authData, error: authError } = await supabase.auth.admin.createUser({
        email: newUserEmail.trim(),
        password: newUserPassword.trim(),
        email_confirm: true,
      });

      if (authError) throw new Error(`Failed to create user: ${authError.message}`);
      if (!authData.user) throw new Error('User creation failed - no user returned');

      // 2. Get or create role_id for the selected role
      const { data: roleData, error: roleError } = await supabase
        .from('roles')
        .select('id')
        .eq('code', newUserRole)
        .maybeSingle();

      if (roleError) throw new Error(`Failed to fetch role: ${roleError.message}`);
      
      const roleId = roleData?.id || crypto.randomUUID();

      // 3. Insert into users_roles_settings table
      const { error: insertError } = await supabase
        .from('users_roles_settings')
        .insert({
          user_id: authData.user.id,
          user_email: newUserEmail.trim(),
          user_display_name: newUserDisplayName.trim() || newUserEmail.trim(),
          user_created_at: new Date().toISOString(),
          role_id: roleId,
          role_code: newUserRole,
          role_name: newUserRole.charAt(0).toUpperCase() + newUserRole.slice(1),
        } as any);

      if (insertError) throw new Error(`Failed to create user record: ${insertError.message}`);

      // Success - reset form and close modal
      setNewUserEmail('');
      setNewUserPassword('');
      setNewUserDisplayName('');
      setNewUserRole('viewer');
      setShowAddUserModal(false);
      
      // Refresh user list
      if (onRefresh) onRefresh();

      alert(`User ${newUserEmail} created successfully with role: ${newUserRole}`);
    } catch (err: any) {
      setCreateError(err.message || 'Failed to create user');
    } finally {
      setIsCreatingUser(false);
    }
  };

  // group settings by user
  type GroupedEntry = { info: UserRoleSetting; settings: UserRoleSetting[] };
  const grouped: Record<string, GroupedEntry> = users.reduce(
    (acc: Record<string, GroupedEntry>, u) => {
    const key = u.userId;
    if (!acc[key]) acc[key] = { info: u, settings: [] };
    if (u.settingKey) acc[key].settings.push(u);
    return acc;
  }, {});

  const entries = Object.values(grouped);

  // Role badge color
  const getRoleBadge = (code: string | null) => {
    switch (code?.toLowerCase()) {
      case 'admin':
        return 'bg-red-500/15 text-red-400 border-red-500/30';
      case 'operator':
        return 'bg-blue-500/15 text-blue-400 border-blue-500/30';
      case 'viewer':
        return 'bg-slate-500/15 text-slate-400 border-slate-500/30';
      default:
        return 'bg-slate-500/15 text-slate-400 border-slate-500/30';
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl shadow-xl overflow-hidden transition-all duration-300">
      {/* Header / Trigger */}
      <div 
        className={`p-6 flex items-center justify-between cursor-pointer hover:bg-slate-800/40 transition-colors border-b ${isOpen ? 'border-slate-800/80' : 'border-transparent'}`}
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center gap-2">
          <Users size={16} className="text-blue-400" />
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            System Access & Roles
          </h3>
        </div>
        
        <div className="flex items-center gap-4">
          {isAdmin && (
            <button 
              onClick={(e) => {
                e.stopPropagation();
                setShowAddUserModal(true);
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 text-xs font-bold uppercase tracking-wider rounded-md border border-blue-500/30 transition-colors"
            >
              <UserPlus size={12} />
              Add User
            </button>
          )}
          <span className="text-[11px] text-slate-500 font-bold uppercase tracking-wider bg-slate-800/50 px-2.5 py-1 rounded-md border border-slate-700/50">
            {entries.length} Active User(s)
          </span>
          <div className="text-slate-500 hover:text-slate-300 transition-colors">
            {isOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </div>
        </div>
      </div>

      {/* Expandable Content */}
      <div 
        className={`grid transition-all duration-300 ease-in-out ${
          isOpen ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
        }`}
      >
        <div className="overflow-hidden">
          <div className="p-6 pt-2 bg-slate-900/30">
            {entries.length === 0 ? (
              <p className="text-xs text-slate-500 py-6 text-center italic font-bold">
                No user data found in the system.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead>
                    <tr className="border-b border-slate-800/80">
                      <th className="pb-3 text-[11px] text-slate-500 font-semibold uppercase tracking-wider">
                        User Profile
                      </th>
                      <th className="pb-3 text-[11px] text-slate-500 font-semibold uppercase tracking-wider">
                        Access Level
                      </th>
                      <th className="pb-3 text-[11px] text-slate-500 font-semibold uppercase tracking-wider">
                        Overrides
                      </th>
                      <th className="pb-3 text-[11px] text-slate-500 font-semibold uppercase tracking-wider text-right">
                        Last Active
                      </th>
                      {isAdmin && (
                        <th className="pb-3 text-[11px] text-slate-500 font-semibold uppercase tracking-wider text-center">
                          Actions
                        </th>
                      )}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50">
                    {entries.map(({ info, settings }) => (
                      <tr key={info.userId} className="group hover:bg-slate-800/20 transition-colors">
                        {/* User info */}
                        <td className="py-4 pr-4">
                          <div className="flex items-center gap-3">
                            <div className="w-9 h-9 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-sm font-bold text-slate-300">
                              {(info.displayName ?? info.email ?? '?').charAt(0).toUpperCase()}
                            </div>
                            <div>
                              <p className="text-sm font-semibold text-slate-200 group-hover:text-blue-400 transition-colors">
                                {info.displayName ?? 'Unnamed User'}
                              </p>
                              <div className="flex items-center gap-1.5 text-xs text-slate-400 mt-0.5 font-medium">
                                <Mail size={12} />
                                <span>{info.email ?? '—'}</span>
                              </div>
                            </div>
                          </div>
                        </td>

                        {/* Role badge */}
                        <td className="py-4 pr-4">
                          <span
                            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-semibold uppercase tracking-wider border border-transparent shadow-sm ${getRoleBadge(info.roleCode)}`}
                          >
                            <ShieldCheck size={12} />
                            {info.roleName ?? info.roleCode ?? '—'}
                          </span>
                        </td>

                        {/* Settings */}
                        <td className="py-4 pr-4">
                          {settings.length > 0 ? (
                            <div className="flex flex-wrap gap-1.5 max-w-[200px]">
                              {settings.map((s, i) => (
                                <span
                                  key={i}
                                  className="text-[11px] font-mono font-medium bg-slate-950 border border-slate-800 text-slate-400 px-2 py-0.5 rounded shadow-sm"
                                >
                                  {s.settingKey}=<span className="text-blue-400">{s.settingValueNumber ?? s.settingValueText ?? '—'}</span>
                                </span>
                              ))}
                            </div>
                          ) : (
                            <span className="text-xs italic text-slate-500">Global Defaults</span>
                          )}
                        </td>

                        {/* Last update */}
                        <td className="py-4 text-right">
                          <div className="flex items-center justify-end gap-1.5 text-xs text-slate-400">
                            <Clock size={14} className="text-slate-500" />
                            <span>
                              {info.settingUpdatedAt
                                ? new Date(info.settingUpdatedAt).toLocaleDateString('vi-VN')
                                : 'Never'}
                            </span>
                          </div>
                        </td>

                        {/* Actions (Admin Only) */}
                        {isAdmin && (
                          <td className="py-4 pl-4 text-center">
                            <button 
                              className="p-1.5 text-slate-500 hover:text-blue-400 hover:bg-blue-500/10 rounded border border-transparent hover:border-blue-500/20 transition-colors"
                              title="Edit User"
                            >
                              <MoreHorizontal size={14} />
                            </button>
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Add User Modal */}
      {showAddUserModal && (
        <>
          <div className="fixed inset-0 bg-slate-950/70 backdrop-blur-sm z-[9998]" onClick={() => setShowAddUserModal(false)} />
          <div className="fixed inset-0 flex items-center justify-center z-[9999] pointer-events-none p-4">
            <div className="bg-slate-900 border border-blue-500/30 rounded-2xl shadow-2xl p-6 max-w-md w-full pointer-events-auto" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-blue-500/10 flex items-center justify-center">
                    <UserPlus size={20} className="text-blue-400" />
                  </div>
                  <div>
                    <h4 className="text-slate-100 font-bold">Add New User</h4>
                    <p className="text-slate-500 text-xs">Create a new user account with role assignment</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowAddUserModal(false)}
                  className="text-slate-500 hover:text-slate-300 transition-colors p-1 rounded hover:bg-slate-800"
                >
                  <X size={18} />
                </button>
              </div>

              {createError && (
                <div className="mb-4 bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-xs text-red-400">
                  {createError}
                </div>
              )}

              <div className="space-y-4 mb-6">
                <div className="space-y-1">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Email Address *</label>
                  <input
                    type="email"
                    value={newUserEmail}
                    onChange={(e) => setNewUserEmail(e.target.value)}
                    placeholder="user@example.com"
                    className="w-full bg-slate-950 border border-slate-700 text-slate-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all placeholder:text-slate-600"
                    autoFocus
                  />
                </div>

                <div className="space-y-1">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Password *</label>
                  <input
                    type="password"
                    value={newUserPassword}
                    onChange={(e) => setNewUserPassword(e.target.value)}
                    placeholder="Minimum 6 characters"
                    className="w-full bg-slate-950 border border-slate-700 text-slate-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all placeholder:text-slate-600"
                  />
                </div>

                <div className="space-y-1">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Display Name</label>
                  <input
                    type="text"
                    value={newUserDisplayName}
                    onChange={(e) => setNewUserDisplayName(e.target.value)}
                    placeholder="John Doe (optional)"
                    className="w-full bg-slate-950 border border-slate-700 text-slate-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all placeholder:text-slate-600"
                  />
                </div>

                <div className="space-y-1">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Role *</label>
                  <select
                    value={newUserRole}
                    onChange={(e) => setNewUserRole(e.target.value as 'admin' | 'operator' | 'viewer')}
                    className="w-full bg-slate-950 border border-slate-700 text-slate-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all"
                  >
                    <option value="viewer">Viewer - Read-only access</option>
                    <option value="operator">Operator - Can manage nodes and alerts</option>
                    <option value="admin">Admin - Full system access</option>
                  </select>
                </div>
              </div>

              <div className="flex justify-end gap-3">
                <button
                  onClick={() => setShowAddUserModal(false)}
                  className="px-4 py-2 rounded-lg text-sm font-bold text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleAddUser}
                  disabled={isCreatingUser || !newUserEmail.trim() || !newUserPassword.trim()}
                  className="px-4 py-2 rounded-lg text-sm font-bold text-white bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-500/20 flex items-center gap-2"
                >
                  {isCreatingUser ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Creating...
                    </>
                  ) : (
                    <>
                      <Check size={14} />
                      Create User
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
