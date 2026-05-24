import React from 'react';
import { Routes, Route, useNavigate, useLocation, Navigate, Outlet } from 'react-router-dom';
import { LanguageProvider } from './core/i18n/LanguageContext';
import {
  Shield,
  LayoutDashboard,
  Activity,
  BrainCircuit,
  Bell,
  ChevronDown,
  Zap,
  RefreshCw,
  Server,
  FileText,
  Settings,
  LogOut
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { DashboardPage } from './modules/dashboard';
import { LiveMonitorPage } from './modules/live-monitor';
import { ThreatAnalyticsPage } from './modules/threat-analytics';
import { ModelInsightsPage } from './modules/model-insights';
import { MLOpsPage } from './modules/mlops';
import { FleetPage } from './modules/fleet-mgmt';
import { ReportsPage } from './modules/reports';
import { SettingsPage } from './modules/settings';
import { NotificationDropdown } from './core/layout/NotificationDropdown';
import { UserDropdown } from './core/layout/UserDropdown';
import { GlobalSearchBar } from './core/layout/GlobalSearchBar';
import { UserAvatar } from './core/components/UserAvatar';
import { AlertHistoryPage } from './modules/alerts-history';
import { ProfilePage } from './modules/profile';
import { ToastContainer } from './modules/fleet-mgmt/view/components/Toast';

// Auth
import { ProtectedRoute } from './core/auth/ProtectedRoute';
import { useAuthContext } from './core/auth/AuthProvider';
import { LoginPage } from './modules/auth';
import { supabase } from './core/lib/supabaseClient';

// --- Components ---

const SidebarItem = ({ icon: Icon, label, path }: { icon: any, label: string, path: string }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const active = location.pathname === path || (path !== '/' && location.pathname.startsWith(path));

  return (
    <div
      onClick={() => navigate(path)}
      className={`flex items-center gap-3 px-4 py-3 cursor-pointer transition-all duration-200 group ${active
        ? 'bg-blue-600/10 border-l-4 border-blue-500 text-blue-400'
        : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
        }`}>
      <Icon size={20} className={active ? 'text-blue-400' : 'text-slate-500 group-hover:text-slate-300'} />
      <span className="font-medium text-sm">{label}</span>
    </div>
  );
};

const SIDEBAR_TRANSLATIONS = {
  en: {
    dashboard: 'Dashboard',
    liveMonitor: 'Live Monitor',
    threatAnalytics: 'Threat Analytics',
    modelInsights: 'Model Insights',
    mlopsCenter: 'MLOps Center',
    edgeFleet: 'Edge Fleet',
    reportsAudit: 'Reports & Audit',
    settings: 'Settings',
    roleUser: 'User',
    roleSocLevel3: 'SOC Level 3',
    adminUser: 'Admin User',
  },
  vi: {
    dashboard: 'Bảng điều khiển',
    liveMonitor: 'Giám sát trực tiếp',
    threatAnalytics: 'Phân tích đe dọa',
    modelInsights: 'Hiển thị mô hình',
    mlopsCenter: 'Trung tâm MLOps',
    edgeFleet: 'Hệ thống nút biên',
    reportsAudit: 'Báo cáo & Kiểm toán',
    settings: 'Cài đặt',
    roleUser: 'Người dùng',
    roleSocLevel3: 'SOC Cấp độ 3',
    adminUser: 'Quản trị viên',
  }
};

const MainLayout = () => {
  const { user, role } = useAuthContext();
  const location = useLocation();
  const navigate = useNavigate();

  const [lang, setLang] = React.useState<'en' | 'vi'>(() => {
    const saved = localStorage.getItem('dashboard_lang');
    return (saved === 'en' || saved === 'vi') ? saved : 'en';
  });

  React.useEffect(() => {
    const handleLangChange = () => {
      const saved = localStorage.getItem('dashboard_lang');
      setLang((saved === 'en' || saved === 'vi') ? saved : 'en');
    };
    window.addEventListener('languageChange', handleLangChange);
    return () => window.removeEventListener('languageChange', handleLangChange);
  }, []);

  const handleSetLang = (newLang: 'en' | 'vi') => {
    localStorage.setItem('dashboard_lang', newLang);
    setLang(newLang);
    window.dispatchEvent(new Event('languageChange'));
  };

  const tSidebar = (key: keyof typeof SIDEBAR_TRANSLATIONS['en']) => {
    return SIDEBAR_TRANSLATIONS[lang][key] || SIDEBAR_TRANSLATIONS['en'][key];
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    navigate('/login', { replace: true });
  };

  // Restricted user: only Dashboard and Live Monitor
  const RESTRICTED_USER_ID = 'f5333890-9353-4db9-9f41-6ad692480e50';
  const isRestrictedUser = user?.id === RESTRICTED_USER_ID;

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 font-sans overflow-hidden">
      <ToastContainer />
      {/* Sidebar */}
      <aside className="w-64 border-r border-slate-800 flex flex-col bg-slate-950 z-20">
        <div className="p-6 flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-transparent flex items-center justify-center shadow-lg shadow-blue-600/20 overflow-hidden">
            <img
              src="/logo_white.png"
              alt="SOC Logo"
              className="w-full h-full object-contain scale-[1.7]"
              referrerPolicy="no-referrer"
            />
          </div>
          <h1 className="font-bold text-lg tracking-tight text-slate-200">Synapse <span className="text-blue-500">IoT</span></h1>
        </div>

        <nav className="flex-1 mt-4 overflow-y-auto custom-scrollbar">
          {/* Always show Dashboard and Live Monitor */}
          <SidebarItem icon={LayoutDashboard} label={tSidebar('dashboard')} path="/" />
          <SidebarItem icon={Activity} label={tSidebar('liveMonitor')} path="/live-monitor" />
          
          {/* Hide other modules for restricted user */}
          {!isRestrictedUser && (
            <>
              <SidebarItem icon={Zap} label={tSidebar('threatAnalytics')} path="/threat-analytics" />
              <SidebarItem icon={BrainCircuit} label={tSidebar('modelInsights')} path="/model-insights" />
              <SidebarItem icon={Settings} label={tSidebar('mlopsCenter')} path="/mlops" />

              <div className="px-4 py-4">
                <div className="h-px bg-slate-800 w-full"></div>
              </div>

              <SidebarItem icon={Server} label={tSidebar('edgeFleet')} path="/fleet" />
              <SidebarItem icon={FileText} label={tSidebar('reportsAudit')} path="/reports" />
              <SidebarItem icon={Settings} label={tSidebar('settings')} path="/settings" />
            </>
          )}
        </nav>

        <div className="p-4 border-t border-slate-800">
          <div className="bg-slate-900 rounded-xl p-4 flex items-center gap-3 relative group cursor-pointer transition-colors hover:bg-slate-800">
            <UserAvatar email={user?.email} size="md" className="group-hover:border-blue-400/50 transition-colors" />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-bold truncate text-slate-200">{user?.email?.split('@')[0] || tSidebar('adminUser')}</p>
              <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">
                {isRestrictedUser ? tSidebar('roleUser') : (role || tSidebar('roleSocLevel3'))}
              </p>
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); handleLogout(); }}
              className="absolute right-4 p-2 rounded-lg bg-slate-800 text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-rose-500/20 hover:text-rose-400"
              title="Sign Out"
            >
              <LogOut size={16} />
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Navbar */}
        <header className="h-16 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-950/50 backdrop-blur-md z-10">
          <div className="flex-1 flex items-center pr-4 min-w-0">
            <GlobalSearchBar />
          </div>

          <div className="flex items-center gap-3 flex-shrink-0">
            {/* Global Language Switcher */}
            <div className="flex items-center bg-slate-900/80 p-0.5 rounded-lg border border-slate-800 mr-2 shadow-inner">
              <button
                onClick={() => handleSetLang('en')}
                className={`px-2.5 py-1 text-[10px] font-extrabold rounded-md cursor-pointer transition-all ${
                  lang === 'en'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                🇬🇧 EN
              </button>
              <button
                onClick={() => handleSetLang('vi')}
                className={`px-2.5 py-1 text-[10px] font-extrabold rounded-md cursor-pointer transition-all ${
                  lang === 'vi'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                🇻🇳 VI
              </button>
            </div>

            <NotificationDropdown />

            <div className="h-8 w-px bg-slate-800 mx-1"></div>

            <UserDropdown />
          </div>
        </header>

        {/* Scrollable Content */}
        <main className="flex-1 overflow-y-auto p-6 custom-scrollbar">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #1e293b;
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #334155;
        }
      `}</style>
    </div>
  );
};

export default function App() {
  return (
    <LanguageProvider>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route element={<ProtectedRoute />}>
          <Route element={<MainLayout />}>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/live-monitor" element={<LiveMonitorPage />} />
            
            {/* Protected routes - redirect restricted user to dashboard */}
            <Route path="/threat-analytics" element={<RestrictedRoute><ThreatAnalyticsPage /></RestrictedRoute>} />
            <Route path="/model-insights" element={<RestrictedRoute><ModelInsightsPage /></RestrictedRoute>} />
            <Route path="/mlops" element={<RestrictedRoute><MLOpsPage /></RestrictedRoute>} />
            <Route path="/fleet" element={<RestrictedRoute><FleetPage /></RestrictedRoute>} />
            <Route path="/alerts" element={<RestrictedRoute><AlertHistoryPage /></RestrictedRoute>} />
            <Route path="/profile" element={<ProfilePage />} />
            <Route path="/reports" element={<RestrictedRoute><ReportsPage /></RestrictedRoute>} />
            <Route path="/settings" element={<RestrictedRoute><SettingsPage /></RestrictedRoute>} />
            
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Route>
      </Routes>
    </LanguageProvider>
  );
}

// Restricted Route Component - blocks access for specific users
const RestrictedRoute = ({ children }: { children: React.ReactNode }) => {
  const { user } = useAuthContext();
  const RESTRICTED_USER_ID = 'f5333890-9353-4db9-9f41-6ad692480e50';
  
  if (user?.id === RESTRICTED_USER_ID) {
    return <Navigate to="/" replace />;
  }
  
  return <>{children}</>;
};
