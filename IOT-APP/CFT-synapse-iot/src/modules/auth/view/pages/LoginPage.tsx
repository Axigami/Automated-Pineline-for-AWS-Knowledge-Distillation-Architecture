import React, { useState, useEffect } from 'react';
import { ShieldCheck, Lock, Mail, ArrowRight, Loader2, Activity } from 'lucide-react';
import { supabase } from '../../../../core/lib/supabaseClient';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuthContext } from '../../../../core/auth/AuthProvider';
import { motion } from 'motion/react';

const DataStreamBackground = () => {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none opacity-20 select-none">
      <style>{`
        @keyframes float-down {
          0% { transform: translateY(-20%); opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { transform: translateY(120vh); opacity: 0; }
        }
        @keyframes scan {
          0% { transform: translateY(-100vh); }
          100% { transform: translateY(100vh); }
        }
      `}</style>
      
      {/* Hex/Grid Pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(14,165,233,0.15)_1px,transparent_1px),linear-gradient(90deg,rgba(14,165,233,0.15)_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_60%_60%_at_50%_50%,black_30%,transparent_100%)] opacity-30"></div>

      {/* Floating Data Columns */}
      {Array.from({ length: 25 }).map((_, i) => (
        <div 
          key={i} 
          className="absolute font-mono text-[10px] text-blue-400/50 whitespace-nowrap"
          style={{
            left: `${Math.random() * 100}%`,
            top: `-20%`,
            animation: `float-down ${Math.random() * 15 + 15}s linear infinite`,
            animationDelay: `-${Math.random() * 20}s`,
            writingMode: 'vertical-rl',
            textOrientation: 'upright',
            textShadow: '0 0 8px rgba(56, 189, 248, 0.4)'
          }}
        >
          {Array.from({ length: 40 }).map(() => 
            Math.random() > 0.5 ? Math.floor(Math.random() * 2).toString() : String.fromCharCode(0x30A0 + Math.random() * 96)
          ).join('')}
        </div>
      ))}

      {/* Cinematic Scanner Line */}
      <div 
        className="absolute inset-0 w-full h-[200px] bg-gradient-to-b from-transparent via-blue-500/10 to-transparent blur-xl pointer-events-none"
        style={{ animation: 'scan 8s cubic-bezier(0.4, 0, 0.2, 1) infinite' }}
      ></div>
    </div>
  );
};

export const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { session, loading } = useAuthContext();
  const from = location.state?.from?.pathname || '/';

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!loading && session) {
      navigate(from, { replace: true });
    }
  }, [session, loading, navigate, from]);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError('System requires both identification factors.');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) {
        throw error;
      }

      if (data.session) {
        navigate(from, { replace: true });
      }
    } catch (err: any) {
      console.error('Login error:', err);
      setError(err.message || 'Authentication sequence failed.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#020617] flex flex-col justify-center items-center p-4 relative overflow-hidden font-sans">
      
      {/* Background Ambience */}
      <DataStreamBackground />
      
      {/* Glowing Orbs */}
      <motion.div 
        animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.5, 0.3] }}
        transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-sky-600/10 rounded-full blur-[120px] pointer-events-none" 
      />
      <motion.div 
        animate={{ scale: [1, 1.3, 1], opacity: [0.2, 0.4, 0.2] }}
        transition={{ duration: 10, repeat: Infinity, ease: "easeInOut", delay: 2 }}
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-[70%] w-[500px] h-[500px] bg-indigo-500/20 rounded-full blur-[100px] pointer-events-none" 
      />

      {/* Main Container - Increased background opacity and border for better contrast */}
      <motion.div 
        initial={{ opacity: 0, y: 40, filter: 'blur(10px)' }}
        animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
        transition={{ duration: 0.8, delay: 0.2, type: "spring", stiffness: 100 }}
        className="w-full max-w-md bg-slate-800/60 backdrop-blur-2xl border border-slate-500/40 rounded-3xl shadow-[0_0_50px_rgba(14,165,233,0.25)] overflow-hidden relative z-10"
      >
        
        {/* Glow Top Border Effect */}
        <div className="absolute top-0 left-0 right-0 h-[3px] bg-gradient-to-r from-transparent via-sky-400 to-transparent opacity-90"></div>
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-48 h-6 bg-sky-500/30 blur-xl"></div>

        <div className="p-10 pt-12">
          {/* Logo & Branding */}
          <div className="flex flex-col items-center justify-center text-center mb-8 mt-[-10px]">
            <motion.div 
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.4, type: "spring", bounce: 0.4 }}
              className="w-24 h-24 rounded-full bg-transparent border border-slate-600/50 shadow-[0_0_40px_rgba(14,165,233,0.3)] flex items-center justify-center mb-6 relative overflow-hidden group"
            >
              <img 
                src="/logo_white.png" 
                alt="Synapse Logo" 
                className="w-full h-full object-contain scale-[1.7]"
                referrerPolicy="no-referrer"
              />
              <div className="absolute inset-0 bg-sky-400 blur-2xl opacity-10 group-hover:opacity-30 transition-opacity"></div>
            </motion.div>
            
            <motion.h1 
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.8 }}
              className="text-3xl tracking-tight font-black text-white mb-2"
            >
              Synapse <span className="text-transparent bg-clip-text bg-gradient-to-r from-sky-400 to-indigo-400">IoT</span>
            </motion.h1>
            
            <motion.p 
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1 }}
              className="text-xs font-black text-sky-400 uppercase tracking-[0.25em]"
            >
              Intelligent Edge Monitor
            </motion.p>
          </div>

          <form onSubmit={handleLogin} className="space-y-6">
            {error && (
              <motion.div 
                initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}
                className="bg-rose-500/10 border border-rose-500/30 rounded-xl p-4 text-sm font-semibold text-rose-400 text-center shadow-[0_0_15px_rgba(244,63,94,0.1)]"
              >
                {error}
              </motion.div>
            )}

            <div className="space-y-5">
              {/* Email Input */}
              <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.5 }} className="relative group">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <Mail size={18} className="text-slate-400 group-focus-within:text-sky-300 transition-colors" />
                </div>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-slate-900/80 border border-slate-600 focus:border-sky-400 focus:ring-4 focus:ring-sky-400/20 rounded-xl py-3.5 pl-11 pr-4 text-white text-sm font-semibold transition-all placeholder:text-slate-400 shadow-inner"
                  placeholder="name@company.com"
                  disabled={isLoading}
                  autoComplete="email"
                />
              </motion.div>

              {/* Password Input */}
              <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.6 }} className="relative group">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <Lock size={18} className="text-slate-400 group-focus-within:text-sky-300 transition-colors" />
                </div>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-slate-900/80 border border-slate-600 focus:border-sky-400 focus:ring-4 focus:ring-sky-400/20 rounded-xl py-3.5 pl-11 pr-4 text-white text-sm font-semibold transition-all placeholder:text-slate-400 shadow-inner"
                  placeholder="••••••••"
                  disabled={isLoading}
                  autoComplete="current-password"
                />
              </motion.div>
            </div>

            {/* Actions */}
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.8 }} className="flex items-center justify-between text-xs font-bold uppercase tracking-wider">
              <label className="flex items-center gap-2 cursor-pointer group">
                <div className="relative flex items-center justify-center w-5 h-5 rounded border border-slate-500 bg-slate-800 overflow-hidden group-hover:border-sky-400 transition-colors">
                  <input type="checkbox" className="absolute opacity-0 w-full h-full cursor-pointer peer" />
                  <div className="w-full h-full bg-sky-500 text-white flex items-center justify-center opacity-0 scale-50 peer-checked:opacity-100 peer-checked:scale-100 transition-all">
                    ✓
                  </div>
                </div>
                <span className="text-slate-300 group-hover:text-white transition-colors">Remember Session</span>
              </label>
              
              <a href="#" className="text-sky-400 hover:text-sky-300 hover:underline transition-colors tooltip-trigger" 
                 title="Contact System Administrator to reset your password.">
                Recover Access
              </a>
            </motion.div>

            <motion.button
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.9 }}
              type="submit"
              disabled={isLoading}
              className="group relative w-full flex justify-center items-center py-4 px-4 rounded-xl text-sm font-black uppercase tracking-widest text-white bg-sky-600 hover:bg-sky-500 focus:outline-none focus:ring-4 focus:ring-sky-500/30 disabled:bg-slate-700 disabled:text-slate-400 disabled:cursor-not-allowed transition-all overflow-hidden shadow-[0_0_20px_rgba(14,165,233,0.5)] hover:shadow-[0_0_30px_rgba(14,165,233,0.7)] disabled:shadow-none"
            >
              <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/30 to-transparent opacity-0 group-hover:opacity-100 -translate-x-full group-hover:translate-x-full transition-transform duration-1000 ease-in-out"></div>
              
              {isLoading ? (
                <div className="flex items-center gap-3">
                  <Loader2 className="animate-spin" size={18} />
                  <span>Authenticating...</span>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <span>Sign In</span>
                  <ArrowRight size={18} className="group-hover:translate-x-1.5 transition-transform" />
                </div>
              )}
            </motion.button>
          </form>
          
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.2 }} className="mt-8 text-center text-[10px] uppercase tracking-[0.2em] font-bold text-slate-400 border-t border-slate-700 pt-6">
            <div className="flex items-center justify-center gap-2 text-sky-400 mb-1">
              <ShieldCheck size={14} /> Intelligent Operations Network
            </div>
            <p className="mt-1 text-slate-400 text-[10px] tracking-widest border-t border-transparent pt-1">
              Platform by <span className="text-white font-black tracking-widest text-[11px]">CRAFTFORGE TEAM</span>
            </p>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
};
