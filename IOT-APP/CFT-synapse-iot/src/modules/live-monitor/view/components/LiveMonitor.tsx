import React, { useState, useMemo, useEffect, useRef } from 'react';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Activity, 
  Zap, 
  ShieldAlert, 
  Cpu, 
  Thermometer, 
  Wifi, 
  ChevronDown, 
  Pause, 
  Play, 
  Volume2, 
  VolumeX,
  Clock,
  ArrowRight,
  Loader2,
  CheckCircle2,
  AlertCircle,
  MemoryStick,
  X,
} from 'lucide-react';
import { useLanguage } from '../../../../core/i18n/LanguageContext';
import type { FlowUIModel, AlertUIModel, EdgeSyncStatus, RadarPoint } from '../../model/types';

// ─────────────── Props Interface ───────────────

interface LiveMonitorProps {
  flows: FlowUIModel[];
  alerts: AlertUIModel[];
  syncStatuses: Record<string, EdgeSyncStatus>;
  radarData: RadarPoint[];
  isPaused: boolean;
  onTogglePause: () => void;
  onVerifyAlert: (alertId: string) => void;
  error: string | null;
  totalFlowsReceived: number;
  /** Thông báo sau khi verify alert ghi DB thành công */
  verifyBanner?: string | null;
  onDismissVerifyBanner?: () => void;
}

// ─────────────── Audio Utility ───────────────

let lastPingTime = 0;
let audioCtx: AudioContext | null = null;

const playAlertSound = () => {
  try {
    const now = Date.now();
    if (now - lastPingTime < 1500) return; // Chỉ phát Ping sạch 1.5 giây một lần dù có bao nhiêu alert tới
    lastPingTime = now;

    if (!audioCtx) {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioContextClass) return;
      audioCtx = new AudioContextClass();
    }

    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    
    // Tạo 2 lớp âm thanh để cho ra tiếng "Glassy Bell" (Chuông thủy tinh) tinh tế
    const osc1 = audioCtx.createOscillator();
    const osc2 = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    osc1.connect(gainNode);
    osc2.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    // Lớp 1: Âm gốc (Sine wave) mượt mà, sâu
    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(1046.50, audioCtx.currentTime); // Nốt C6

    // Lớp 2: Âm họa ba (Triangle wave) tạo độ lanh lảnh của tiếng chuông
    osc2.type = 'triangle';
    osc2.frequency.setValueAtTime(2093.00, audioCtx.currentTime); // Nốt C7 (Octave)

    // Hiệu ứng "Gõ chuông": Đánh mạnh tức khắc (Attack) và ngân vang tắt dần (Decay)
    gainNode.gain.setValueAtTime(0, audioCtx.currentTime);
    gainNode.gain.linearRampToValueAtTime(0.4, audioCtx.currentTime + 0.02);
    gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.8);

    osc1.start(audioCtx.currentTime);
    osc2.start(audioCtx.currentTime);
    osc1.stop(audioCtx.currentTime + 0.8);
    osc2.stop(audioCtx.currentTime + 0.8);
  } catch (err) {
    console.error("Audio playback error:", err);
  }
};

// ─────────────── Sub-components ───────────────

interface SyncStatusProps {
  syncStatus: EdgeSyncStatus | null;
}

const SyncStatus: React.FC<SyncStatusProps> = ({ syncStatus }) => {
  const { t } = useLanguage();
  return (
    <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-xl p-4 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest flex items-center gap-2">
          <Wifi size={12} className="text-emerald-500" /> {t('liveMonitor', 'syncTitle')}
        </h4>
        <div className="flex items-center gap-1">
          <div className={`w-1.5 h-1.5 rounded-full animate-pulse ${
            syncStatus?.isOnline ? 'bg-emerald-500' : 'bg-rose-500'
          }`} />
          <span className={`text-xs font-bold uppercase ${
            syncStatus?.isOnline ? 'text-emerald-500' : 'text-rose-500'
          }`}>
            {syncStatus ? (syncStatus.isOnline ? t('liveMonitor', 'online') : t('liveMonitor', 'offline')) : t('liveMonitor', 'connecting')}
          </span>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-4">
        {/* CPU */}
        <div className="space-y-1">
          <div className="flex items-center gap-1 text-slate-500">
            <Cpu size={12} />
            <span className="text-xs font-bold uppercase">{t('liveMonitor', 'cpu')}</span>
          </div>
          <p className="text-sm font-mono text-slate-200">
            {syncStatus?.cpu != null ? `${syncStatus.cpu.toFixed(1)}%` : '—'}
          </p>
          {syncStatus?.cpu != null && (
            <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  syncStatus.cpu > 80 ? 'bg-rose-500' : syncStatus.cpu > 60 ? 'bg-amber-500' : 'bg-emerald-500'
                }`}
                style={{ width: `${Math.min(syncStatus.cpu, 100)}%` }}
              />
            </div>
          )}
        </div>

        {/* Temperature */}
        <div className="space-y-1">
          <div className="flex items-center gap-1 text-slate-500">
            <Thermometer size={12} />
            <span className="text-xs font-bold uppercase">{t('liveMonitor', 'temp')}</span>
          </div>
          <p className="text-sm font-mono text-slate-200">
            {syncStatus?.temperature != null ? `${syncStatus.temperature.toFixed(1)}°C` : '—'}
          </p>
          {syncStatus?.temperature != null && (
            <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  syncStatus.temperature > 75 ? 'bg-rose-500' : syncStatus.temperature > 60 ? 'bg-amber-500' : 'bg-blue-500'
                }`}
                style={{ width: `${Math.min((syncStatus.temperature / 100) * 100, 100)}%` }}
              />
            </div>
          )}
        </div>

        {/* Latency */}
        <div className="space-y-1">
          <div className="flex items-center gap-1 text-slate-500">
            <Clock size={12} />
            <span className="text-xs font-bold uppercase">{t('liveMonitor', 'latency')}</span>
          </div>
          <p className="text-sm font-mono text-slate-200">
            {syncStatus?.latencyMs != null ? `${syncStatus.latencyMs}ms` : '—'}
          </p>
          {syncStatus?.ram != null && (
            <div className="space-y-0.5">
              <div className="flex items-center gap-1 text-slate-600">
                <MemoryStick size={10} />
                <span className="text-[10px] font-bold uppercase">{t('liveMonitor', 'ram')} {syncStatus.ram.toFixed(0)}%</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Node code + last seen */}
      {syncStatus && (
        <div className="mt-4 pt-3 border-t border-slate-800/60 flex justify-between items-center">
          <span className="text-xs font-mono text-slate-400">{syncStatus.nodeCode}</span>
          {syncStatus.lastSeenAt && (
            <span className="text-xs font-mono text-slate-500">
              {t('liveMonitor', 'lastSeen')} {syncStatus.lastSeenAt}
            </span>
          )}
        </div>
      )}
    </div>
  );
};

// ─── AlertCard – render 1 cảnh báo ───
interface AlertCardProps {
  alert: AlertUIModel;
  onVerify: (id: string) => void;
}

const AlertCard: React.FC<AlertCardProps> = ({ alert, onVerify }) => {
  const { t } = useLanguage();
  return (
    <motion.div 
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      layout
      className={`bg-slate-900/80 backdrop-blur-md border rounded-xl p-4 mb-4 transition-all ${
        alert.severity === 'critical' ? 'border-rose-500/30' : 'border-amber-500/30'
      }`}
    >
      <div className="flex justify-between items-start mb-3">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg flex items-center justify-center ${alert.severity === 'critical' ? 'bg-rose-500/20 text-rose-500' : 'bg-amber-500/20 text-amber-500'}`}>
            <ShieldAlert size={18} className={alert.severity === 'critical' ? 'animate-pulse' : ''} />
          </div>
          <div className="flex flex-col justify-center">
            <h5 className={`text-[15px] font-bold uppercase tracking-tight leading-none ${alert.severity === 'critical' ? 'text-rose-400' : 'text-amber-400'}`}>
              {alert.threat}
            </h5>
            <p className="text-xs text-slate-500 font-mono leading-none mt-1.5">{alert.timeDisplay || alert.time}</p>
          </div>
        </div>
        <div className="text-right flex flex-col justify-center">
          <span className="text-[11px] font-bold text-slate-500 uppercase tracking-widest leading-none">{t('liveMonitor', 'confidence')}</span>
          <p className="text-[15px] font-mono font-bold text-slate-300 leading-none mt-1.5">
            {alert.confidence != null
              ? `${(alert.confidence * 100).toFixed(1)}%`
              : '—'}
          </p>
        </div>
      </div>

      {/* Source IP if available */}
      {alert.srcIp && (
        <p className="text-xs font-mono text-slate-500 mb-3">
          <span className="text-slate-600 font-sans font-semibold mr-1">src:</span> {alert.srcIp}
        </p>
      )}

      <div className="mb-5">
        <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3">{t('liveMonitor', 'triggerSeq')}</p>
        <div className="flex gap-1 h-8 items-end">
          {alert.sequence.map((val, i) => (
            <div 
              key={i} 
              className={`flex-1 rounded-t-sm transition-all ${val > 0.7 ? 'bg-rose-500' : 'bg-slate-700'}`}
              style={{ height: `${val * 100}%` }}
            />
          ))}
        </div>
      </div>

      {alert.status === 'pending' && (
        <button 
          onClick={() => onVerify(alert.id)}
          className="w-full py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm font-bold uppercase tracking-widest rounded-lg transition-all flex items-center justify-center gap-2 cursor-pointer shadow-md"
        >
          <Zap size={14} /> {t('liveMonitor', 'verifyBtn')}
        </button>
      )}

      {alert.status === 'verifying' && (
        <div className="w-full py-2.5 bg-slate-800 text-slate-400 text-sm font-bold uppercase tracking-widest rounded-lg flex items-center justify-center gap-2 border border-slate-700">
          <Loader2 size={14} className="animate-spin" /> {t('liveMonitor', 'verifying')}
        </div>
      )}

      {alert.status === 'confirmed' && (
        <div className="w-full py-2.5 bg-rose-500/20 border border-rose-500/50 text-rose-400 text-sm font-bold uppercase tracking-widest rounded-lg flex items-center justify-center gap-2 shadow-inner">
          <AlertCircle size={14} /> {alert.verdict || t('liveMonitor', 'confirmedThreat')}
        </div>
      )}

      {alert.status === 'false-positive' && (
        <div className="w-full py-2.5 bg-emerald-500/20 border border-emerald-500/50 text-emerald-400 text-sm font-bold uppercase tracking-widest rounded-lg flex items-center justify-center gap-2 shadow-inner">
          <CheckCircle2 size={14} /> {t('liveMonitor', 'falsePositive')}
        </div>
      )}
    </motion.div>
  );
};

// ─────────────────────────────── Main Component ───────────────

export const LiveMonitor: React.FC<LiveMonitorProps> = ({
  flows = [],
  alerts = [],
  syncStatuses = {},
  radarData = [],
  isPaused,
  onTogglePause,
  onVerifyAlert,
  error,
  totalFlowsReceived,
  verifyBanner,
  onDismissVerifyBanner,
}) => {
  const { lang, t } = useLanguage();
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [selectedNodeId, setSelectedNodeId] = useState('All Nodes');
  const prevAlertCount = useRef(alerts.length);
  const lastAlertSourceIpRef = useRef<string | null>(null); // Track last alert source IP
  
  // Track last data update time
  const [lastUpdateTime, setLastUpdateTime] = useState<Date>(new Date());
  const [timeSinceUpdate, setTimeSinceUpdate] = useState<string>('just now');
  
  // Update last update time when new data arrives
  useEffect(() => {
    if (flows.length > 0 || alerts.length > 0) {
      setLastUpdateTime(new Date());
    }
  }, [flows.length, alerts.length]);
  
  // Update "time ago" display every second
  useEffect(() => {
    const updateTimeAgo = () => {
      const now = new Date();
      const diffSeconds = Math.floor((now.getTime() - lastUpdateTime.getTime()) / 1000);
      
      if (diffSeconds < 5) {
        setTimeSinceUpdate(t('liveMonitor', 'justNow'));
      } else if (diffSeconds < 60) {
        setTimeSinceUpdate(t('liveMonitor', 'secondsAgo', diffSeconds));
      } else if (diffSeconds < 3600) {
        const mins = Math.floor(diffSeconds / 60);
        setTimeSinceUpdate(t('liveMonitor', 'minutesAgo', mins));
      } else {
        const hours = Math.floor(diffSeconds / 3600);
        setTimeSinceUpdate(t('liveMonitor', 'hoursAgo', hours));
      }
    };
    
    updateTimeAgo();
    const interval = setInterval(updateTimeAgo, 1000);
    return () => clearInterval(interval);
  }, [lastUpdateTime, lang]);

  // Trigger audio alert when a new threat arrives with DIFFERENT source IP
  useEffect(() => {
    if (!isPaused && soundEnabled && alerts.length > prevAlertCount.current) {
      // Get the newest alert (first in array since alerts are sorted newest first)
      const newestAlert = alerts[0];
      const newSourceIp = newestAlert?.srcIp || null;
      
      // Only play sound if source IP is different from last alert
      if (newSourceIp !== lastAlertSourceIpRef.current) {
        console.log(`[LiveMonitor] New alert from different source IP: ${newSourceIp} (previous: ${lastAlertSourceIpRef.current})`);
        playAlertSound();
        lastAlertSourceIpRef.current = newSourceIp;
      } else {
        console.log(`[LiveMonitor] Alert from same source IP (${newSourceIp}) - sound suppressed`);
      }
    }
    prevAlertCount.current = alerts.length;
  }, [alerts.length, soundEnabled, isPaused]);

  const availableNodes = useMemo(() => (Object.values(syncStatuses) as EdgeSyncStatus[]).sort((a, b) => a.nodeCode.localeCompare(b.nodeCode)), [syncStatuses]);

  const filteredFlows = useMemo(() => {
    if (selectedNodeId === 'All Nodes') return flows;
    const selectedNodeInfo = syncStatuses[selectedNodeId];
    return flows.filter(f => f.nodeId === selectedNodeId || (selectedNodeInfo && f.homeId === selectedNodeInfo.homeId));
  }, [flows, selectedNodeId, syncStatuses]);

  const filteredAlerts = useMemo(() => {
    if (selectedNodeId === 'All Nodes') return alerts;
    const selectedNodeInfo = syncStatuses[selectedNodeId];
    return alerts.filter(a => a.nodeId === selectedNodeId || (selectedNodeInfo && a.homeId === selectedNodeInfo.homeId));
  }, [alerts, selectedNodeId, syncStatuses]);

  // Anomaly count từ flows thực
  const anomalyCount = useMemo(
    () => filteredFlows.filter((f) => f.isAnomaly).length,
    [filteredFlows]
  );
  
  // Custom Radar data based on filtered flows
  // Chỉ dùng dữ liệu thực từ DB – không mock/random
  const computedRadarData = useMemo(() => {
    if (selectedNodeId === 'All Nodes') return radarData;

    const recent = [...filteredFlows].slice(0, 30).reverse();
    const len = recent.length;
    const paddedLen = 30 - len;

    // Padding bên trái = 0 (chưa có dữ liệu cho node này)
    const padPoints: RadarPoint[] = Array.from({ length: paddedLen }, (_, i) => ({
      time: i,
      score: 0,
    }));

    // Điểm thực từ flows DB: ưu tiên anomaly_score → confidence → 0
    const histPoints: RadarPoint[] = recent.map((f, i) => ({
      time: paddedLen + i,
      score: f.anomalyScore !== null
        ? f.anomalyScore
        : f.isAnomaly
        ? (f.confidence ?? 0.5)
        : 0,
    }));

    return [...padPoints, ...histPoints];
  }, [radarData, filteredFlows, selectedNodeId]);

  // Compute average sync status if All Nodes, else specific node
  const computedSyncStatus = useMemo(() => {
    if (selectedNodeId === 'All Nodes') {
      const statuses = Object.values(syncStatuses) as EdgeSyncStatus[];
      if (statuses.length === 0) return null;
      let onlineCount = 0;
      let cpuSum = 0, cpuCount = 0;
      let ramSum = 0, ramCount = 0;
      let tempSum = 0, tempCount = 0;
      let latSum = 0, latCount = 0;
      
      statuses.forEach(s => {
        if (s.isOnline) onlineCount++;
        if (s.cpu != null) { cpuSum += s.cpu; cpuCount++; }
        if (s.ram != null) { ramSum += s.ram; ramCount++; }
        if (s.temperature != null) { tempSum += s.temperature; tempCount++; }
        if (s.latencyMs != null) { latSum += s.latencyMs; latCount++; }
      });
      return {
        id: 'all',
        nodeCode: lang === 'en' ? 'All Nodes (Avg)' : 'Tất cả các nút (TB)',
        isOnline: onlineCount > 0,
        cpu: cpuCount > 0 ? cpuSum / cpuCount : null,
        ram: ramCount > 0 ? ramSum / ramCount : null,
        temperature: tempCount > 0 ? tempSum / tempCount : null,
        latencyMs: latCount > 0 ? Math.round(latSum / latCount) : null,
        lastSeenAt: null
      };
    }
    return syncStatuses[selectedNodeId] || null;
  }, [selectedNodeId, syncStatuses, lang]);

  return (
    <div className="space-y-6">
      {verifyBanner && (
        <div className="flex items-start justify-between gap-3 rounded-xl border border-emerald-500/40 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-200 -mx-6">
          <p className="pt-0.5 leading-relaxed">{verifyBanner}</p>
          {onDismissVerifyBanner && (
            <button
              type="button"
              onClick={onDismissVerifyBanner}
              className="shrink-0 rounded-lg p-1 text-emerald-400/80 hover:bg-emerald-500/20 hover:text-emerald-200"
              aria-label="Close"
            >
              <X size={18} />
            </button>
          )}
        </div>
      )}

      {/* Sticky Header */}
      <div className="sticky top-0 z-30 bg-slate-950/80 backdrop-blur-xl border-b border-slate-800 -mx-6 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div className="relative">
            <select 
              value={selectedNodeId}
              onChange={(e) => setSelectedNodeId(e.target.value)}
              className="appearance-none bg-slate-900 border border-slate-700 text-slate-100 text-sm font-bold py-2.5 pl-4 pr-10 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/50 cursor-pointer uppercase tracking-widest shadow-lg"
            >
              <option value="All Nodes">{t('liveMonitor', 'allNodes')}</option>
              {availableNodes.map(node => (
                <option key={node.id} value={node.id}>{node.nodeCode}</option>
              ))}
            </select>
            <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
          </div>
          <div className="h-6 w-px bg-slate-800" />
          
          {/* Connection Status with Animation */}
          <div className="flex items-center gap-2">
            <div className="relative">
              <div className={`w-2.5 h-2.5 rounded-full ${computedSyncStatus?.isOnline ? 'bg-emerald-500' : 'bg-rose-500'} ${!isPaused && computedSyncStatus?.isOnline ? 'animate-pulse' : ''}`} />
              {!isPaused && computedSyncStatus?.isOnline && (
                <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-emerald-500 animate-ping" />
              )}
            </div>
            <span className="text-sm font-bold text-slate-300 uppercase tracking-widest">
              {isPaused ? t('liveMonitor', 'paused') : computedSyncStatus?.isOnline ? t('liveMonitor', 'liveStreaming') : computedSyncStatus ? t('liveMonitor', 'offline') : t('liveMonitor', 'connecting')}
            </span>
          </div>
          
          {/* Last Update Time */}
          {!isPaused && (
            <>
              <div className="h-6 w-px bg-slate-800" />
              <div className="flex items-center gap-2">
                <Clock size={12} className="text-slate-500" />
                <span className="text-sm font-medium text-slate-400">
                  {t('liveMonitor', 'lastUpdate')} <span className="font-bold text-slate-300">{timeSinceUpdate}</span>
                </span>
              </div>
            </>
          )}
          
          {/* Realtime stats pill */}
          {totalFlowsReceived > 0 && (
            <>
              <div className="h-6 w-px bg-slate-800" />
              <span className="text-sm font-mono text-slate-400">
                {t('liveMonitor', 'flowsReceived', totalFlowsReceived)}
              </span>
              {anomalyCount > 0 && (
                <span className="text-sm font-mono text-rose-400 font-bold animate-pulse">
                  {t('liveMonitor', 'anomaliesDetected', anomalyCount)}
                </span>
              )}
            </>
          )}
        </div>

        <div className="flex items-center gap-3">
          {/* Error badge */}
          {error && (
            <span className="text-[10px] font-bold text-rose-400 bg-rose-500/10 border border-rose-500/30 px-2 py-1 rounded">
              {error}
            </span>
          )}

          <button 
            onClick={() => setSoundEnabled(!soundEnabled)}
            className={`p-2 rounded-lg border transition-all cursor-pointer ${soundEnabled ? 'bg-blue-600/10 border-blue-500/50 text-blue-400' : 'bg-slate-900 border-slate-800 text-slate-500'}`}
          >
            {soundEnabled ? <Volume2 size={18} /> : <VolumeX size={18} />}
          </button>
          <button 
            onClick={onTogglePause}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg border font-bold text-sm uppercase tracking-widest transition-all cursor-pointer ${
              isPaused 
                ? 'bg-emerald-600/10 border-emerald-500/50 text-emerald-400 hover:bg-emerald-600/20' 
                : 'bg-slate-900 border-slate-700 text-slate-300 hover:bg-slate-800'
            }`}
          >
            {isPaused ? <><Play size={16} /> {t('liveMonitor', 'resumeStream')}</> : <><Pause size={16} /> {t('liveMonitor', 'pauseStream')}</>}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-10 gap-6 items-stretch h-[calc(100vh-140px)] min-h-[800px]">
        {/* Left Section (70%) */}
        <div className="lg:col-span-7 flex flex-col h-full space-y-6">
          {/* Active Sequence Radar */}
          <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-xl p-6 shadow-xl shrink-0">
            {(() => {
              // Tính toán score hiện tại và màu gradient thích ứng
              const currentScore = computedRadarData.length > 0
                ? computedRadarData[computedRadarData.length - 1].score
                : 0;
              const isHighAlert   = currentScore > 0.7;
              const isMedAlert    = currentScore > 0.5 && !isHighAlert;
              const gradientId    = isHighAlert ? 'radarGradientHigh' : isMedAlert ? 'radarGradientAmber' : 'radarGradient';
              const strokeColor   = isHighAlert ? '#f43f5e' : isMedAlert ? '#f59e0b' : '#3b82f6';
              const scoreBadgeClass = isHighAlert
                ? 'bg-rose-500/20 border border-rose-500/50 text-rose-400'
                : isMedAlert
                ? 'bg-amber-500/20 border border-amber-500/50 text-amber-400'
                : 'bg-emerald-500/20 border border-emerald-500/50 text-emerald-400';

              return (
                <>
                  <div className="flex justify-between items-center mb-6">
                    <div>
                      <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                        <Activity size={16} className={isHighAlert ? 'text-rose-400 animate-pulse' : 'text-blue-400'} /> {t('liveMonitor', 'activeRadar')}
                      </h3>
                      <p className="text-xs text-slate-500 font-bold uppercase tracking-widest mt-1">
                        {t('liveMonitor', 'radarSub')}
                      </p>
                    </div>
                    <div className="flex items-center gap-4">
                      {/* Current score badge */}
                      <div className={`px-3 py-1.5 rounded-lg font-mono text-sm font-bold ${scoreBadgeClass}`}>
                        {(currentScore * 100).toFixed(1)}%
                      </div>
                      <div className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full`} style={{ background: strokeColor }} />
                        <span className="text-xs font-bold text-slate-400 uppercase">Score</span>
                      </div>
                      {/* Live indicator */}
                      {!isPaused && (
                        <div className="flex items-center gap-1">
                          <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-ping" />
                          <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-widest">Live</span>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="h-[250px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={computedRadarData}>
                        <defs>
                          {/* Gradient xanh: bình thường */}
                          <linearGradient id="radarGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.35}/>
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                          </linearGradient>
                          {/* Gradient vàng: cảnh báo trung bình */}
                          <linearGradient id="radarGradientAmber" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.35}/>
                            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                          </linearGradient>
                          {/* Gradient đỏ: nguy hiểm cao */}
                          <linearGradient id="radarGradientHigh" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.45}/>
                            <stop offset="95%" stopColor="#f43f5e" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                        <XAxis dataKey="time" hide />
                        <YAxis stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', color: '#f8fafc' }}
                          labelStyle={{ display: 'none' }}
                          formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Anomaly Score']}
                        />
                        {/* Threshold line tại 0.7 – vùng nguy hiểm */}
                        <ReferenceLine
                          y={0.7}
                          stroke="#f43f5e"
                          strokeDasharray="4 4"
                          strokeOpacity={0.6}
                          label={{ value: 'Threat', fill: '#f43f5e', fontSize: 10, fontWeight: 700, dx: -6 }}
                        />
                        {/* Threshold line tại 0.5 – vùng cảnh báo */}
                        <ReferenceLine
                          y={0.5}
                          stroke="#f59e0b"
                          strokeDasharray="4 4"
                          strokeOpacity={0.4}
                          label={{ value: 'Warn', fill: '#f59e0b', fontSize: 10, fontWeight: 700, dx: -6 }}
                        />
                        <Area 
                          type="monotone" 
                          dataKey="score" 
                          stroke={strokeColor}
                          fillOpacity={1} 
                          fill={`url(#${gradientId})`}
                          strokeWidth={2} 
                          isAnimationActive={false}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </>
              );
            })()}
          </div>

          {/* Dual-Stream Raw Data Table */}
          <div className="flex flex-col flex-1 bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-xl overflow-hidden shadow-xl min-h-0">
            <div className="p-4 border-b border-slate-800 bg-slate-900/50 flex justify-between items-center shrink-0">
              <h3 className="text-xs font-bold text-slate-200 uppercase tracking-widest flex items-center gap-2">
                <ArrowRight size={14} className="text-blue-400" /> {t('liveMonitor', 'rawTableTitle')}
              </h3>
              <div className="flex gap-4">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-slate-700 animate-pulse" />
                  <span className="text-xs font-bold text-slate-500 uppercase">{t('liveMonitor', 'telemetry')}</span>
                </div>
              </div>
            </div>
            
            <div className="flex-1 overflow-y-auto custom-scrollbar">
              <table className="w-full text-left border-collapse">
                <thead className="sticky top-0 bg-slate-900 z-10 shadow-sm">
                  <tr>
                    <th className="px-4 py-3 text-sm font-bold text-slate-500 uppercase tracking-widest border-b border-slate-800">{t('liveMonitor', 'tblTime')}</th>
                    <th className="px-4 py-3 text-sm font-bold text-slate-500 uppercase tracking-widest border-b border-slate-800">{t('liveMonitor', 'tblProtocol')}</th>
                    <th className="px-4 py-3 text-sm font-bold text-slate-500 uppercase tracking-widest border-b border-slate-800">{t('liveMonitor', 'tblSrcPort')}</th>
                    <th className="px-4 py-3 text-sm font-bold text-slate-500 uppercase tracking-widest border-b border-slate-800">{t('liveMonitor', 'tblDstPort')}</th>
                    <th className="px-4 py-3 text-sm font-bold text-slate-500 uppercase tracking-widest border-b border-slate-800">{t('liveMonitor', 'tblDuration')}</th>
                    <th className="px-4 py-3 text-sm font-bold text-slate-500 uppercase tracking-widest border-b border-slate-800">{t('liveMonitor', 'tblBytes')}</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-sm">
                  <AnimatePresence initial={false}>
                    {filteredFlows.length === 0 ? (
                      <tr>
                        <td colSpan={6} className="px-4 py-12 text-center">
                          <div className="flex flex-col items-center gap-2 text-slate-600">
                            <Activity size={24} className="opacity-30" />
                            <span className="text-xs font-bold uppercase tracking-widest">
                              {isPaused ? t('liveMonitor', 'streamPaused') : t('liveMonitor', 'noDataNode')}
                            </span>
                          </div>
                        </td>
                      </tr>
                    ) : (
                      filteredFlows.map((flow, index) => (
                        <motion.tr 
                          key={`${flow.id}-${flow.timestamp}-${index}`}
                          initial={{ opacity: 0, y: -10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className={`border-b border-slate-800/50 transition-colors hover:bg-slate-800/30 ${
                            flow.isAnomaly ? 'bg-rose-500/5' : ''
                          }`}
                        >
                          <td className="px-4 py-2 text-slate-400">{flow.timestampDisplay || flow.timestamp}</td>
                          <td className="px-4 py-2">
                            <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                              flow.protocol === 'TCP' ? 'bg-blue-500/10 text-blue-400' : 
                              flow.protocol === 'UDP' ? 'bg-amber-500/10 text-amber-400' :
                              flow.protocol === 'ICMP' ? 'bg-purple-500/10 text-purple-400' :
                              'bg-slate-800 text-slate-400'
                            }`}>
                              {flow.protocol}
                            </span>
                          </td>
                          <td className="px-4 py-2 text-slate-300">{flow.srcPort}</td>
                          <td className="px-4 py-2 text-slate-300">{flow.dstPort}</td>
                          <td className="px-4 py-2 text-slate-400">{flow.duration.toFixed(3)}s</td>
                          <td className="px-4 py-2 text-slate-400">
                            {flow.bytes > 1024
                              ? `${(flow.bytes / 1024).toFixed(1)} KB`
                              : `${flow.bytes} B`
                            }
                          </td>
                        </motion.tr>
                      ))
                    )}
                  </AnimatePresence>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Right Section (30%) */}
        <div className="lg:col-span-3 flex flex-col h-full space-y-6">
          {/* Edge-to-Cloud Sync */}
          <SyncStatus syncStatus={computedSyncStatus} />

          {/* Live Alert Stack */}
          <div className="flex flex-col flex-1 bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-xl p-6 shadow-xl overflow-hidden min-h-0">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                <ShieldAlert size={16} className="text-rose-400" /> {t('liveMonitor', 'liveAlertStack')}
              </h3>
              <span className="text-xs font-bold text-slate-500 uppercase tracking-widest bg-slate-950 px-2 py-1 rounded border border-slate-800">
                {t('liveMonitor', 'activeAlertsCount', filteredAlerts.filter(a => a.status === 'pending' || a.status === 'verifying').length)}
              </span>
            </div>

            <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
              <AnimatePresence mode="popLayout">
                {filteredAlerts.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-slate-600 space-y-2 min-h-[200px]">
                    <Activity size={32} className="opacity-20" />
                    <p className="text-xs font-bold uppercase tracking-widest">{t('liveMonitor', 'noActiveThreats')}</p>
                  </div>
                ) : (
                  filteredAlerts.map((alert, index) => (
                    <AlertCard key={`${alert.id}-${alert.time}-${index}`} alert={alert} onVerify={onVerifyAlert} />
                  ))
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
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

export default LiveMonitor;
