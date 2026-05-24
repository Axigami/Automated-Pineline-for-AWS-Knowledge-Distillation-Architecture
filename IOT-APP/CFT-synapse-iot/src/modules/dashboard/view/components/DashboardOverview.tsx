import React, { useState, useMemo, useEffect } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, LineChart, Line, PieChart, Pie, Legend
} from 'recharts';
import { motion, AnimatePresence } from 'motion/react';
import {
  Server, AlertTriangle, CloudLightning, TrendingUp, ShieldCheck,
  Cpu, Database, Thermometer, Microscope, X, Binary, Fingerprint,
  Zap, Activity, Brain, RefreshCw, Wifi, WifiOff, Clock, CheckCircle2,
} from 'lucide-react';
import type {
  TelemetrySummary, AlertUIModel, TrafficSeriesPoint,
  EdgeNodeRow, AttackDistPoint, ModelStats, HomeInfo,
} from '../../model/types';

const CHART_COLORS = [
  '#3b82f6', '#f43f5e', '#10b981', '#f59e0b',
  '#8b5cf6', '#06b6d4', '#ec4899', '#f97316'
];

// ─────────────────────────────────────────────────────────────
// Props
// ─────────────────────────────────────────────────────────────
interface DashboardOverviewProps {
  summary: TelemetrySummary | null;
  recentAlerts: AlertUIModel[];
  trafficSeries: TrafficSeriesPoint[];
  homes: HomeInfo[];
  edgeNodes: EdgeNodeRow[];
  attackDist: AttackDistPoint[];
  modelStats: ModelStats | null;
  networkFlowMetrics: any[];
  isLoading: boolean;
  isAdmin: boolean;
  onRefresh: () => void;
  enableNode?: (ipAddress: string, action: 'unblock' | 'isolate') => Promise<{ success: boolean; error?: string }>;
  dismissAlert?: (alertId: string) => Promise<{ success: boolean; error?: string }>;
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────
function calcSecurityScore(summary: TelemetrySummary | null): number {
  if (!summary || summary.alertsToday === 0) return 100;
  const raw = Math.round((1 - summary.criticalAlerts / Math.max(summary.alertsToday, 1)) * 100);
  return Math.min(100, Math.max(0, raw));
}

function severityColor(sev: AlertUIModel['severity']) {
  if (sev === 'high') return 'text-rose-400';
  if (sev === 'medium') return 'text-amber-400';
  return 'text-slate-400';
}

function statusBadge(status: string) {
  const s = status.toLowerCase();
  if (s === 'verified') return 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30';
  if (s === 'pending') return 'bg-amber-500/15 text-amber-400 border-amber-500/30';
  if (s === 'dismissed') return 'bg-slate-500/15 text-slate-400 border-slate-500/30';
  return 'bg-blue-500/15 text-blue-400 border-blue-500/30';
}

const TRANSLATIONS = {
  en: {
    // Alert status
    pending: 'Pending',
    verified: 'Verified',
    dismissed: 'Dismissed',
    statusPending: 'Pending',
    statusVerified: 'Verified',
    statusDismissed: 'Dismissed',
    
    // Attacker/target details
    attacker: 'Attacker:',
    target: 'Target:',
    location: 'Location:',
    device: 'Device:',
    severity: 'Severity:',
    severityCritical: '⚠️ Critical',
    severityMedium: '⚡ Medium',
    severityLow: '✓ Low',
    analyzing: 'Analyzing...',
    verifyViaCloud: 'Verify via Cloud',

    // Forensic view
    forensicTitleAdmin: 'Sequence Forensic View',
    forensicTitleUser: 'Alert Details',
    src: 'SRC:',
    node: 'NODE:',
    time: 'TIME:',
    locationCaps: 'LOCATION:',
    deviceCaps: 'DEVICE:',
    timeCaps: 'TIME:',
    seqAnalysis: 'Sequence Analysis',
    stepsCount: (n: number) => `(${n} steps)`,
    scoreLabel: 'Score:',
    packetStep: 'Packet / Step',
    anomalyScore: 'Anomaly Score',
    status: 'Status',
    normal: 'NORMAL',
    triggerEvent: (step: number) => `Trigger Event Detected at Step ${step}`,
    cloudVerification: 'Cloud Verification',
    cloudAiVerification: 'Cloud AI Verification',
    confirmedMalicious: 'CONFIRMED MALICIOUS',
    anomalyConfirmed: 'ANOMALY CONFIRMED',
    confidenceScore: 'Confidence Score',
    predictionConfidence: 'Prediction Confidence',
    autoResponseProfile: 'Automated Response Profile',
    recommendedAction: 'Recommended Action',
    adminRecommendation: (label: string, ip: string) => `Sequence analysis confirms a high probability of ${label} attack from ${ip}. Suggest preemptive node isolation.`,
    userRecommendation: (label: string, loc: string, dev: string) => `The system detected an anomaly labeled ${label} at location ${loc} on device ${dev}. We recommend isolating this device to ensure home network safety.`,

    // Actions
    btnEnableNode: 'Enable Node',
    btnEnabling: 'Enabling...',
    btnIsolateNode: 'Isolate Node',
    btnIsolating: 'Isolating...',
    btnReenableDevice: 'Re-enable Device',
    btnIsolateDevice: 'Isolate Device',
    btnDismissFlag: 'Dismiss Flag',
    btnDismissing: 'Dismissing...',
    btnDismissAlert: 'Dismiss Alert',

    // Dialog messages
    alertNodeEnabledSuccessAdmin: '✅ Node enabled successfully. IP unblocked.',
    alertNodeEnabledSuccessUser: '✅ Device unblocked and enabled successfully.',
    alertNodeEnabledFailAdmin: '❌ Failed to enable node',
    alertNodeEnabledFailUser: '❌ Failed to enable device',
    alertNodeIsolatedSuccessAdmin: '✅ Node isolated successfully.',
    alertNodeIsolatedSuccessUser: '✅ Device isolated successfully.',
    alertNodeIsolatedFailAdmin: '❌ Failed to isolate node',
    alertNodeIsolatedFailUser: '❌ Failed to isolate device',
    alertDismissedSuccessAdmin: '✅ Alert dismissed successfully',
    alertDismissedSuccessUser: '✅ Alert dismissed successfully',
    alertDismissedFailAdmin: '❌ Failed to dismiss alert',
    alertDismissedFailUser: '❌ Failed to dismiss alert',

    // Main labels
    live: 'LIVE',
    updated: 'Updated',
    justNow: 'just now',
    secondsAgo: (s: number) => `${s}s ago`,
    minutesAgo: (m: number) => `${m}m ago`,
    hoursAgo: (h: number) => `${h}h ago`,
    autoRefresh: 'Auto-refresh: 30s',
    refreshing: 'Refreshing...',
    refreshNow: 'Refresh Now',

    // KPI Cards
    totalDevices: 'Total Devices',
    activeAlerts: 'Active Alerts',
    cloudAiStatus: 'Cloud AI Status',
    avgConfidence: 'Avg Confidence',
    securityScore: 'Security Score',
    devices: 'Devices',
    safetyScore: 'Security Score',

    // KPI Card Sub-values
    onlinePct: (pct: number) => `${pct}% Online`,
    criticalAlerts: (c: number) => `${c} Critical`,
    loading: 'Loading',
    last20Alerts: 'Last 20 Alerts',
    noData: 'No Data',
    excellent: 'Excellent',
    good: 'Good',
    atRisk: 'At Risk',
    onlinePctVi: (pct: number) => `${pct}% Online`,

    // Simplified layout texts
    onlineDevicesPct: (pct: number) => `${pct}% Active`,
    safetyVeryGood: 'Excellent',
    safetyGood: 'Good',
    safetyAttention: 'Attention Needed',

    // Alert List & Items
    latestAlerts: 'Latest Alerts',
    liveAlertFeed: 'Live Alert Feed',
    noAlertsFound: 'No alerts found',
    viewAllIncidents: 'VIEW ALL INCIDENTS →',
    viewAllAlerts: 'VIEW ALL ALERTS →',

    // Network Flow Metrics
    flowTitleAdmin: 'Network Flow Metrics',
    flowTitleUser: 'Network Flow & Activity',
    flowSubAdmin: 'Traffic Volume & Anomalies — Last 24h',
    flowSubUser: 'Traffic & Anomalies — Last 24h',
    flowsLabel: 'Total Connections',
    anomaliesLabel: 'Anomalies',
    noFlowData: 'No network flow data in last 24h',
    waitingFlowData: 'Waiting for network flow data...',
    waitingDeviceData: 'Waiting for data from devices...',
    flowSummaryAdmin: (flows: number, anoms: number) => `Total records: ${flows} flows, ${anoms} anomalies`,
    flowSummaryUser: (flows: number, anoms: number) => `Total: ${flows} connections, ${anoms} anomalous connections`,
    dataVolumeLabel: 'Data Volume',

    // Bottom Row (Admin-only)
    attackDistTitle: 'Attack Distribution',
    last24h: 'Last 24h',
    totalRecorded: 'Total Recorded',
    topVector: 'Top Vector',
    noAttackData: 'No attack data in 24h',
    edgeNodeFleet: 'Edge Node Fleet',
    nodesOnline: (online: number, total: number) => `${online}/${total} Online`,
    noEdgeNodes: 'No edge nodes found',
    cpu: 'CPU',
    ram: 'RAM',
    temp: 'TEMP',
    aiPerfPanel: 'AI Performance Panel',
    f1Score: 'F1-Score',
    accuracy: 'Accuracy',
    precision: 'Precision',
    recall: 'Recall',
    fpRate: 'FP Rate',
    latency: 'Latency',
    throughput: 'Throughput',
    noModelData: 'No model data available'
  },
  vi: {
    // Alert status
    pending: 'Đang chờ',
    verified: 'Đã xác nhận',
    dismissed: 'Đã bỏ qua',
    statusPending: 'Đang chờ',
    statusVerified: 'Đã xác nhận',
    statusDismissed: 'Đã bỏ qua',

    // Attacker/target details
    attacker: 'Nguồn tấn công:',
    target: 'Mục tiêu:',
    location: 'Vị trí:',
    device: 'Thiết bị:',
    severity: 'Mức độ:',
    severityCritical: '⚠️ Nghiêm trọng',
    severityMedium: '⚡ Trung bình',
    severityLow: '✓ Thấp',
    analyzing: 'Đang phân tích...',
    verifyViaCloud: 'Xác minh qua Cloud',

    // Forensic view
    forensicTitleAdmin: 'Phân tích chuỗi gói tin',
    forensicTitleUser: 'Chi tiết Cảnh báo',
    src: 'Nguồn:',
    node: 'Nút:',
    time: 'Thời gian:',
    locationCaps: 'VỊ TRÍ:',
    deviceCaps: 'THIẾT BỊ:',
    timeCaps: 'THỜI GIAN:',
    seqAnalysis: 'Phân tích chuỗi sự kiện',
    stepsCount: (n: number) => `(${n} bước)`,
    scoreLabel: 'Điểm số:',
    packetStep: 'Gói tin / Bước',
    anomalyScore: 'Điểm bất thường',
    status: 'Trạng thái',
    normal: 'BÌNH THƯỜNG',
    triggerEvent: (step: number) => `Phát hiện sự kiện kích hoạt tại Bước ${step}`,
    cloudVerification: 'Xác thực đám mây',
    cloudAiVerification: 'Xác thực từ hệ thống Cloud AI',
    confirmedMalicious: 'XÁC NHẬN ĐỘC HẠI',
    anomalyConfirmed: 'XÁC NHẬN CÓ BẤT THƯỜNG',
    confidenceScore: 'Điểm số tin cậy',
    predictionConfidence: 'Độ tin cậy dự đoán',
    autoResponseProfile: 'Hồ sơ phản hồi tự động',
    recommendedAction: 'Đề xuất xử lý tự động',
    adminRecommendation: (label: string, ip: string) => `Phân tích chuỗi xác nhận khả năng cao xảy ra cuộc tấn công ${label} từ ${ip}. Đề xuất cô lập nút mạng phủ đầu.`,
    userRecommendation: (label: string, loc: string, dev: string) => `Hệ thống phát hiện hoạt động bất thường mang nhãn hiệu ${label} tại ${loc} trên thiết bị ${dev}. Đề xuất cô lập thiết bị này để đảm bảo an toàn cho mạng gia đình.`,

    // Actions
    btnEnableNode: 'Kích hoạt nút',
    btnEnabling: 'Đang kích hoạt...',
    btnIsolateNode: 'Cô lập nút',
    btnIsolating: 'Đang cô lập...',
    btnReenableDevice: 'Kích hoạt lại',
    btnIsolateDevice: 'Cô lập thiết bị',
    btnDismissFlag: 'Bỏ qua cảnh báo',
    btnDismissing: 'Đang bỏ qua...',
    btnDismissAlert: 'Bỏ qua cảnh báo',

    // Dialog messages
    alertNodeEnabledSuccessAdmin: '✅ Nút mạng đã được kích hoạt thành công. Đã mở chặn IP.',
    alertNodeEnabledSuccessUser: '✅ Đã mở chặn và kích hoạt thiết bị thành công.',
    alertNodeEnabledFailAdmin: '❌ Kích hoạt nút mạng thất bại',
    alertNodeEnabledFailUser: '❌ Không thể kích hoạt thiết bị',
    alertNodeIsolatedSuccessAdmin: '✅ Đã cô lập nút mạng thành công.',
    alertNodeIsolatedSuccessUser: '✅ Đã cô lập thiết bị thành công.',
    alertNodeIsolatedFailAdmin: '❌ Cô lập nút mạng thất bại',
    alertNodeIsolatedFailUser: '❌ Không thể cô lập thiết bị',
    alertDismissedSuccessAdmin: '✅ Đã bỏ qua cảnh báo thành công.',
    alertDismissedSuccessUser: '✅ Đã bỏ qua cảnh báo thành công.',
    alertDismissedFailAdmin: '❌ Bỏ qua cảnh báo thất bại',
    alertDismissedFailUser: '❌ Không thể bỏ qua cảnh báo',

    // Main labels
    live: 'TRỰC TIẾP',
    updated: 'Đã cập nhật',
    justNow: 'vừa xong',
    secondsAgo: (s: number) => `${s} giây trước`,
    minutesAgo: (m: number) => `${m} phút trước`,
    hoursAgo: (h: number) => `${h} giờ trước`,
    autoRefresh: 'Tự động tải lại: 30 giây',
    refreshing: 'Đang tải lại...',
    refreshNow: 'Tải lại ngay',

    // KPI Cards
    totalDevices: 'Tổng thiết bị',
    activeAlerts: 'Cảnh báo đang hoạt động',
    cloudAiStatus: 'Trạng thái Cloud AI',
    avgConfidence: 'Độ tin cậy TB',
    securityScore: 'Điểm an toàn',
    devices: 'Thiết bị',
    safetyScore: 'Điểm an toàn',

    // KPI Card Sub-values
    onlinePct: (pct: number) => `${pct}% Đang hoạt động`,
    criticalAlerts: (c: number) => `${c} Nghiêm trọng`,
    loading: 'Đang tải',
    last20Alerts: '20 cảnh báo gần nhất',
    noData: 'Không có dữ liệu',
    excellent: 'Rất tốt',
    good: 'Tốt',
    atRisk: 'Cần chú ý',
    onlinePctVi: (pct: number) => `${pct}% Đang hoạt động`,

    // Simplified layout texts
    onlineDevicesPct: (pct: number) => `${pct}% Đang hoạt động`,
    safetyVeryGood: 'Rất tốt',
    safetyGood: 'Tốt',
    safetyAttention: 'Cần chú ý',

    // Alert List & Items
    latestAlerts: 'Cảnh báo mới nhất',
    liveAlertFeed: 'Cảnh báo trực tiếp',
    noAlertsFound: 'Không có cảnh báo',
    viewAllIncidents: 'XEM TẤT CẢ SỰ CỐ →',
    viewAllAlerts: 'XEM TẤT CẢ CẢNH BÁO →',

    // Network Flow Metrics
    flowTitleAdmin: 'Số liệu lưu lượng mạng',
    flowTitleUser: 'Lưu lượng & Hoạt động mạng',
    flowSubAdmin: 'Khối lượng lưu lượng & Bất thường — 24 giờ qua',
    flowSubUser: 'Lưu lượng & Bất thường — 24 giờ qua',
    flowsLabel: 'Tổng kết nối',
    anomaliesLabel: 'Bất thường',
    noFlowData: 'Không có dữ liệu lưu lượng mạng trong 24 giờ qua',
    waitingFlowData: 'Đang chờ nhận dữ liệu từ các thiết bị...',
    waitingDeviceData: 'Đang chờ nhận dữ liệu từ các thiết bị...',
    flowSummaryAdmin: (flows: number, anoms: number) => `Tổng cộng: ${flows} kết nối, ${anoms} kết nối bất thường`,
    flowSummaryUser: (flows: number, anoms: number) => `Tổng cộng: ${flows} kết nối, ${anoms} kết nối bất thường`,
    dataVolumeLabel: 'Dung lượng dữ liệu',

    // Bottom Row (Admin-only)
    attackDistTitle: 'Phân bổ tấn công',
    last24h: '24 giờ qua',
    totalRecorded: 'Tổng ghi nhận',
    topVector: 'Mối đe dọa chính',
    noAttackData: 'Không có dữ liệu tấn công trong 24 giờ qua',
    edgeNodeFleet: 'Hệ thống nút mạng biên',
    nodesOnline: (online: number, total: number) => `${online}/${total} Hoạt động`,
    noEdgeNodes: 'Không tìm thấy nút mạng biên',
    cpu: 'CPU',
    ram: 'RAM',
    temp: 'Nhiệt độ',
    aiPerfPanel: 'Bảng hiệu suất AI',
    f1Score: 'Chỉ số F1-Score',
    accuracy: 'Độ chính xác',
    precision: 'Độ chuẩn xác',
    recall: 'Độ nhạy',
    fpRate: 'Tỉ lệ dương tính giả',
    latency: 'Độ trễ',
    throughput: 'Thông lượng',
    noModelData: 'Không có dữ liệu mô hình'
  }
};

const getTranslation = (lang: 'en' | 'vi', key: keyof typeof TRANSLATIONS['en'], ...args: any[]) => {
  const template = TRANSLATIONS[lang]?.[key] || TRANSLATIONS['en'][key];
  if (typeof template === 'function') {
    return (template as any)(...args);
  }
  return template;
};


// ─────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────
const KPICard: React.FC<{
  icon: React.ElementType;
  title: string;
  value: string | number;
  subValue: string;
  colorClass: string;
  loading?: boolean;
}> = ({ icon: Icon, title, value, subValue, colorClass, loading }) => (
  <motion.div
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm hover:border-slate-700 transition-all"
  >
    <div className="flex justify-between items-start mb-4">
      <Icon size={22} className={colorClass} />
      <span className={`text-[10px] font-bold tracking-wide uppercase px-2 py-0.5 rounded-full bg-slate-800/50 ${colorClass}`}>
        {subValue}
      </span>
    </div>
    <h3 className="text-slate-400 text-xs font-bold uppercase tracking-widest mb-2">{title}</h3>
    {loading ? (
      <div className="h-10 w-20 bg-slate-800 rounded animate-pulse" />
    ) : (
      <p className="text-4xl font-black text-slate-100 tracking-tighter drop-shadow-sm">{value}</p>
    )}
  </motion.div>
);

const AlertItem: React.FC<{
  alert: AlertUIModel;
  onClick: () => void;
  isSelected: boolean;
  isAdmin: boolean;
  lang: 'en' | 'vi';
}> = ({ alert, onClick, isSelected, isAdmin, lang }) => {
  const t = (key: keyof typeof TRANSLATIONS['en'], ...args: any[]) => getTranslation(lang, key, ...args);

  return (
    <div
      onClick={onClick}
      className={`cursor-pointer border rounded-xl p-4 mb-3 transition-all group relative overflow-hidden ${isSelected
          ? 'bg-blue-600/10 border-blue-500/50 shadow-[0_0_15px_rgba(37,99,235,0.2)]'
          : 'bg-slate-900 border-slate-800 hover:border-rose-500/30'
        }`}
    >
      {isSelected && <div className="absolute left-0 top-0 bottom-0 w-1 bg-blue-500" />}
      <div className="flex justify-between text-slate-500 mb-2 font-mono items-center">
        <span className="text-[11px] font-medium">{alert.time}</span>
        <span className={`text-[9px] font-black uppercase tracking-widest px-2 py-0.5 rounded border ${statusBadge(alert.status)}`}>
          {alert.status === 'pending' ? t('pending') : alert.status === 'verified' ? t('verified') : alert.status === 'dismissed' ? t('dismissed') : alert.status}
        </span>
      </div>
      <div className="flex items-center justify-between mb-3">
        <span className={`font-black text-base tracking-tight ${isSelected ? 'text-blue-400' : severityColor(alert.severity)}`}>
          {alert.label}
        </span>
        <span className="text-slate-400 text-xs font-bold font-mono bg-slate-950/50 px-1.5 py-0.5 rounded">{alert.confidencePct}</span>
      </div>
      
      {/* Attack Details - User-friendly: show location & device instead of raw IPs */}
      <div className="space-y-2 mb-4 bg-slate-950/50 p-3 rounded-lg border border-slate-800/50">
        {isAdmin ? (
          /* Admin view: show raw IPs */
          <>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-bold text-rose-400 uppercase tracking-widest">{t('attacker')}</span>
              <span className="text-xs font-mono text-rose-300 bg-rose-500/10 px-2 py-0.5 rounded border border-rose-500/20">{alert.srcIp}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-bold text-amber-400 uppercase tracking-widest">{t('target')}</span>
              <span className="text-xs font-mono text-amber-300 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">
                {alert.targetIp}
              </span>
            </div>
          </>
        ) : (
          /* User view: show friendly location & device names */
          <>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-widest">{t('location')}</span>
              <span className="text-xs font-semibold text-cyan-300 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                {alert.locationName || (lang === 'en' ? 'Unknown' : 'Không xác định')}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-bold text-amber-400 uppercase tracking-widest">{t('device')}</span>
              <span className="text-xs font-semibold text-amber-300 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">
                {alert.deviceName || (lang === 'en' ? 'Unknown' : 'Không xác định')}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-bold text-rose-400 uppercase tracking-widest">{t('severity')}</span>
              <span className={`text-xs font-semibold px-2 py-0.5 rounded border ${
                alert.severity === 'high' 
                  ? 'text-rose-300 bg-rose-500/10 border-rose-500/20' 
                  : alert.severity === 'medium' 
                    ? 'text-amber-300 bg-amber-500/10 border-amber-500/20' 
                    : 'text-slate-300 bg-slate-500/10 border-slate-500/20'
              }`}>
                {alert.severity === 'high' ? t('severityCritical') : alert.severity === 'medium' ? t('severityMedium') : t('severityLow')}
              </span>
            </div>
          </>
        )}
      </div>
      
      <div className={`w-full py-2 rounded-lg text-xs font-bold flex items-center justify-center gap-2 transition-colors ${isSelected ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-300 group-hover:bg-slate-700'
        }`}>
        {isSelected ? <Microscope size={14} /> : <CloudLightning size={14} />}
        {isSelected ? t('analyzing') : t('verifyViaCloud')}
      </div>
    </div>
  );
};

const ForensicView: React.FC<{
  alert: AlertUIModel;
  onClose: () => void;
  isAdmin: boolean;
  lang: 'en' | 'vi';
  enableNode?: (ipAddress: string, action: 'unblock' | 'isolate') => Promise<{ success: boolean; error?: string }>;
  dismissAlert?: (alertId: string) => Promise<{ success: boolean; error?: string }>;
}> = ({ alert, onClose, isAdmin, lang, enableNode, dismissAlert }) => {
  const t = (key: keyof typeof TRANSLATIONS['en'], ...args: any[]) => getTranslation(lang, key, ...args);
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Determine if this is a DDoS attack
  const isDDoS = alert.label.toLowerCase().includes('ddos');
  
  // Handle button 1 click (Enable Node for DDoS, Isolate Node for others)
  // Both send to IoT Core topic 'Ddos' with different actions
  const handleButton1Click = async () => {
    if (!enableNode) return;
    
    setIsProcessing(true);
    try {
      if (isDDoS) {
        // Enable node (unblock IP) - send action: "unblock"
        console.log('[ForensicView] Enabling node (unblock) - IP:', alert.srcIp);
        const result = await enableNode(alert.srcIp, 'unblock');
        
        if (result.success) {
          window.alert(isAdmin ? t('alertNodeEnabledSuccessAdmin') : t('alertNodeEnabledSuccessUser'));
          onClose();
        } else {
          window.alert(isAdmin 
            ? `${t('alertNodeEnabledFailAdmin')}: ${result.error || 'Unknown error'}` 
            : `${t('alertNodeEnabledFailUser')}: ${result.error || 'Lỗi không xác định'}`
          );
        }
      } else {
        // Isolate node - send action: "isolate"
        console.log('[ForensicView] Isolating node - IP:', alert.srcIp);
        const result = await enableNode(alert.srcIp, 'isolate');
        
        if (result.success) {
          window.alert(isAdmin ? t('alertNodeIsolatedSuccessAdmin') : t('alertNodeIsolatedSuccessUser'));
          onClose();
        } else {
          window.alert(isAdmin 
            ? `${t('alertNodeIsolatedFailAdmin')}: ${result.error || 'Unknown error'}` 
            : `${t('alertNodeIsolatedFailUser')}: ${result.error || 'Lỗi không xác định'}`
          );
        }
      }
    } catch (error) {
      console.error('[ForensicView] Error:', error);
      window.alert(`❌ Error: ${error}`);
    } finally {
      setIsProcessing(false);
    }
  };
  
  // Handle button 2 click (Dismiss alert)
  const handleDismissClick = async () => {
    if (!dismissAlert) return;
    
    setIsProcessing(true);
    try {
      console.log('[ForensicView] Dismissing alert:', alert.id);
      const result = await dismissAlert(alert.id);
      
      if (result.success) {
        window.alert(isAdmin ? t('alertDismissedSuccessAdmin') : t('alertDismissedSuccessUser'));
        onClose();
      } else {
        window.alert(isAdmin 
          ? `${t('alertDismissedFailAdmin')}: ${result.error || 'Unknown error'}` 
          : `${t('alertDismissedFailUser')}: ${result.error || 'Lỗi không xác định'}`
        );
      }
    } catch (error) {
      console.error('[ForensicView] Error dismissing alert:', error);
      window.alert(`❌ Error: ${error}`);
    } finally {
      setIsProcessing(false);
    }
  };
  
  // Build forensic rows từ seqSteps + seqValues, fallback to generated if empty
  const forensicRows = useMemo(() => {
    const steps = alert.seqSteps.length > 0
      ? alert.seqSteps
      : Array.from({ length: 10 }, (_, i) => `pkt_${String(i + 1).padStart(2, '0')}`);
    const vals = alert.seqValues.length > 0
      ? alert.seqValues
      : Array.from({ length: steps.length }, (_, i) => Math.min(1, 0.3 + i * 0.07));

    return steps.map((step, idx) => {
      const rawValue = vals[idx];
      // Ensure value is a number
      const numValue = typeof rawValue === 'number' ? rawValue : parseFloat(String(rawValue)) || 0;
      
      return {
        step,
        value: numValue,
        isAnomaly: idx === steps.length - 1 && numValue > 0.8,
      };
    });
  }, [alert.seqSteps, alert.seqValues]);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-2xl h-full flex flex-col relative overflow-hidden"
    >
      <div className="absolute inset-0 bg-[linear-gradient(rgba(30,41,59,0.5)_1px,transparent_1px),linear-gradient(90deg,rgba(30,41,59,0.5)_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_at_center,black_40%,transparent_100%)] pointer-events-none opacity-20" />

      <div className="flex justify-between items-start mb-8 relative z-10">
        <div className="flex items-center gap-5">
          <div className="p-4 bg-rose-500/10 rounded-2xl border border-rose-500/20 shadow-inner">
            <Microscope className="text-rose-400" size={28} />
          </div>
          <div>
            <h3 className="text-2xl font-black text-slate-100 tracking-tight flex items-center gap-3">
              {isAdmin ? t('forensicTitleAdmin') : t('forensicTitleUser')}
              <span className="text-[10px] bg-rose-500/20 border border-rose-500/30 text-rose-400 px-3 py-1 rounded-full uppercase tracking-widest font-black">
                {alert.label}
              </span>
            </h3>
            {isAdmin ? (
              <p className="text-sm text-slate-400 font-medium mt-1.5 flex gap-4">
                <span><strong className="text-slate-500">{t('src')}</strong> <span className="font-mono">{alert.srcIp}</span></span>
                <span><strong className="text-slate-500">{t('node')}</strong> <span className="font-mono">{alert.source}</span></span>
                <span><strong className="text-slate-500">{t('time')}</strong> <span className="font-mono">{alert.time}</span></span>
              </p>
            ) : (
              <p className="text-sm text-slate-400 font-medium mt-1.5 flex flex-wrap gap-x-6 gap-y-1.5">
                <span><strong className="text-slate-500">{t('locationCaps')}</strong> <span className="text-cyan-400 font-bold">{alert.locationName || (lang === 'en' ? 'Unknown' : 'Không xác định')}</span></span>
                <span><strong className="text-slate-500">{t('deviceCaps')}</strong> <span className="text-amber-400 font-bold">{alert.deviceName || (lang === 'en' ? 'Unknown' : 'Không xác định')}</span></span>
                <span><strong className="text-slate-500">{t('timeCaps')}</strong> <span className="font-mono">{alert.time}</span></span>
              </p>
            )}
          </div>
        </div>
        <button onClick={onClose} className="p-2.5 bg-slate-900 border border-slate-800 hover:bg-slate-800 hover:border-slate-700 rounded-xl transition-all text-slate-400 hover:text-white shadow-sm">
          <X size={20} />
        </button>
      </div>

      <div className={`grid grid-cols-1 ${isAdmin ? 'xl:grid-cols-2' : ''} gap-6 flex-1 relative z-10`}>
        {/* Sequence Table */}
        {isAdmin && (
          <div className="flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-base font-bold text-slate-100 flex items-center gap-2">
                <Binary size={18} className="text-blue-400" /> {t('seqAnalysis')}
                <span className="text-xs font-bold text-slate-500 uppercase tracking-widest ml-2 hidden sm:inline-block">{t('stepsCount', forensicRows.length)}</span>
              </h4>
              <span className="text-xs font-bold bg-slate-900 border border-slate-800 px-3 py-1 rounded-lg text-slate-300 font-mono">{t('scoreLabel')} {alert.confidencePct}</span>
            </div>
            <div className="bg-slate-950/50 rounded-xl border border-slate-800 overflow-hidden flex-1 shadow-inner">
              <table className="w-full text-left">
                <thead>
                  <tr className="text-[10px] font-bold text-slate-400 uppercase tracking-widest border-b border-slate-800 bg-slate-900">
                    <th className="py-3 px-4 w-12 text-center">#</th>
                    <th className="py-3 px-4">{t('packetStep')}</th>
                    <th className="py-3 px-4">{t('anomalyScore')}</th>
                    <th className="py-3 px-4 text-center">{t('status')}</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-sm text-slate-300">
                  {forensicRows.map((row, idx) => (
                    <tr
                       key={idx}
                       className={`border-b border-slate-800/30 transition-colors ${row.isAnomaly ? 'bg-rose-500/10 text-rose-300 border-rose-500/20' : 'hover:bg-slate-900/30'
                         }`}
                    >
                      <td className="py-3 px-4 text-center text-slate-500 text-xs font-bold">{idx + 1}</td>
                      <td className="py-3 px-4 font-medium">{row.step}</td>
                      <td className={`py-3 px-4 font-bold ${row.isAnomaly ? 'text-rose-400' : 'text-slate-400'}`}>
                        {row.value.toFixed(3)}
                      </td>
                      <td className="py-3 px-4 text-center">
                        <span className={`px-2 py-1 rounded border text-[10px] font-black uppercase tracking-widest ${row.isAnomaly
                            ? 'bg-rose-500/20 text-rose-400 border-rose-500/30'
                            : 'bg-slate-800 text-slate-400 border-slate-700'
                          }`}>
                          {row.isAnomaly ? alert.label.toUpperCase() : t('normal')}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="p-3 bg-rose-500/5 border-t border-rose-500/10 text-xs text-rose-400 flex items-center gap-2 justify-center font-medium">
                <AlertTriangle size={14} />
                {t('triggerEvent', forensicRows.length)}
              </div>
            </div>
          </div>
        )}

        {/* Verification Verdict & Actions */}
        <div className="flex flex-col">
          {/* Title Outside */}
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-base font-bold text-slate-100 flex items-center gap-2">
              <ShieldCheck size={18} className="text-emerald-500" /> {isAdmin ? t('cloudVerification') : t('cloudAiVerification')}
            </h4>
          </div>

          <div className="flex flex-col gap-6 flex-1">
            {/* Verdict Card */}
            <div className="bg-gradient-to-br from-slate-900 to-slate-950 rounded-xl border border-slate-800 p-8 relative overflow-hidden">
              <div className="absolute top-0 right-0 p-8 opacity-5">
                <ShieldCheck size={160} className="text-emerald-500" />
              </div>

              <div className="relative z-10 flex flex-col gap-8">
                {/* Row 1: Details & Score */}
                <div className="flex flex-col 2xl:flex-row 2xl:items-center justify-between gap-6">
                  <div>
                    <div className="text-3xl font-bold text-slate-200 tracking-tight flex items-center gap-3">
                      {alert.label}
                    </div>
                    <div className="flex flex-wrap items-center gap-3 mt-3">
                      <span className="text-xs font-medium text-slate-400 bg-slate-800 border border-slate-700 px-2 py-1 rounded">
                        Class_ID: {alert.label.length % 5 + 1}
                      </span>
                      <span className="text-[11px] text-rose-400 font-mono flex items-center gap-1 bg-rose-500/10 px-2 py-1 rounded border border-rose-500/20">
                        <AlertTriangle size={12} /> {isAdmin ? t('confirmedMalicious') : t('anomalyConfirmed')}
                      </span>
                    </div>
                  </div>

                  <div className="text-left 2xl:text-right">
                    <div className="text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-rose-500">
                      {alert.confidencePct}
                    </div>
                    <p className="text-[10px] text-slate-500 uppercase font-bold mt-1 tracking-widest">
                      {isAdmin ? t('confidenceScore') : t('predictionConfidence')}
                    </p>
                  </div>
                </div>

                {/* Row 2: Progress Bar */}
                <div>
                  <div className="relative h-3 bg-slate-800/80 rounded-full overflow-hidden border border-slate-700">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${alert.confidenceVal * 100}%` }}
                      transition={{ duration: 1, ease: 'easeOut' }}
                      className="absolute top-0 left-0 h-full bg-gradient-to-r from-blue-500 via-purple-500 to-rose-500"
                    >
                      {/* CSS Shimmer strip overlay */}
                      <div className="absolute inset-0 opacity-20 bg-[linear-gradient(45deg,transparent_25%,rgba(255,255,255,1)_50%,transparent_75%)] bg-[length:2rem_2rem] animate-[spin_3s_linear_infinite]" style={{ animation: 'shimmer 2s linear infinite', backgroundSize: '1rem 1rem' }} />
                    </motion.div>
                  </div>
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-2">
                    <span>0%</span><span>50%</span><span>100%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Recommended Actions */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 flex flex-col justify-between flex-1 relative z-10 shadow-inner">
              <div>
                <h4 className="text-[11px] font-bold text-emerald-500 uppercase tracking-widest flex items-center gap-2 mb-4">
                  <ShieldCheck size={14} /> {isAdmin ? t('autoResponseProfile') : t('recommendedAction')}
                </h4>
                <p className="text-base text-slate-300 leading-relaxed font-medium">
                  {isAdmin ? (
                    <>{t('adminRecommendation', alert.label, alert.srcIp)}</>
                  ) : (
                    <>{t('userRecommendation', alert.label, alert.locationName || (lang === 'en' ? 'Unknown location' : 'vị trí không xác định'), alert.deviceName || (lang === 'en' ? 'Unknown device' : 'thiết bị không xác định'))}</>
                  )}
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-3 w-full mt-6 pt-6 border-t border-slate-800">
                <button 
                  onClick={handleButton1Click}
                  disabled={isProcessing}
                  className={`flex-1 text-white text-xs font-bold py-3 rounded-lg transition-colors flex items-center justify-center gap-2 border shadow-lg disabled:opacity-50 disabled:cursor-not-allowed ${
                    isDDoS 
                      ? 'bg-emerald-600 hover:bg-emerald-500 border-emerald-500 shadow-emerald-500/20' 
                      : 'bg-blue-600 hover:bg-blue-500 border-blue-500 shadow-blue-500/20'
                  }`}
                >
                  {isDDoS ? (
                    <>
                      <Wifi size={14} /> {isAdmin ? (isProcessing ? t('btnEnabling') : t('btnEnableNode')) : (isProcessing ? t('btnEnabling') : t('btnReenableDevice'))}
                    </>
                  ) : (
                    <>
                      <WifiOff size={14} /> {isAdmin ? (isProcessing ? t('btnIsolating') : t('btnIsolateNode')) : (isProcessing ? t('btnIsolating') : t('btnIsolateDevice'))}
                    </>
                  )}
                </button>
                <button 
                  onClick={handleDismissClick}
                  disabled={isProcessing}
                  className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-bold py-3 rounded-lg transition-colors border border-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isAdmin ? (isProcessing ? t('btnDismissing') : t('btnDismissFlag')) : (isProcessing ? t('btnDismissing') : t('btnDismissAlert'))}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

// ─────────────────────────────────────────────────────────────
// Main Export
// ─────────────────────────────────────────────────────────────
export const DashboardOverview: React.FC<DashboardOverviewProps> = ({
  summary,
  recentAlerts,
  trafficSeries,
  homes,
  edgeNodes,
  attackDist,
  modelStats,
  networkFlowMetrics,
  isLoading,
  isAdmin,
  onRefresh,
  enableNode,
  dismissAlert,
}) => {
  const [selectedAlertId, setSelectedAlertId] = useState<string | null>(null);
  const selectedAlert = recentAlerts.find(a => a.id === selectedAlertId);
  
  const [lang, setLang] = useState<'en' | 'vi'>(() => {
    const saved = localStorage.getItem('dashboard_lang');
    return (saved === 'en' || saved === 'vi') ? saved : 'en';
  });

  useEffect(() => {
    const handleLangChange = () => {
      const saved = localStorage.getItem('dashboard_lang');
      setLang((saved === 'en' || saved === 'vi') ? saved : 'en');
    };
    window.addEventListener('languageChange', handleLangChange);
    return () => window.removeEventListener('languageChange', handleLangChange);
  }, []);

  const t = (key: keyof typeof TRANSLATIONS['en'], ...args: any[]) => getTranslation(lang, key, ...args);

  // Debug: Log isAdmin value
  useEffect(() => {
    console.log('[DashboardOverview] isAdmin:', isAdmin);
    console.log('[DashboardOverview] edgeNodes count:', edgeNodes.length);
    console.log('[DashboardOverview] homes count:', homes.length);
  }, [isAdmin, edgeNodes.length, homes.length]);
  
  // Debug: Log network flow metrics
  useEffect(() => {
    console.log('[DashboardOverview] networkFlowMetrics:', networkFlowMetrics);
    console.log('[DashboardOverview] networkFlowMetrics length:', networkFlowMetrics?.length);
    if (networkFlowMetrics && networkFlowMetrics.length > 0) {
      console.log('[DashboardOverview] Sample metric:', networkFlowMetrics[0]);
      console.log('[DashboardOverview] Total flows:', networkFlowMetrics.reduce((sum, m) => sum + (m.flows || 0), 0));
    }
  }, [networkFlowMetrics]);
  
  // Track last refresh time
  const [lastRefreshTime, setLastRefreshTime] = useState<Date>(new Date());
  const [timeSinceRefresh, setTimeSinceRefresh] = useState<string>('just now');
  
  // Update last refresh time when data changes
  useEffect(() => {
    if (!isLoading) {
      setLastRefreshTime(new Date());
    }
  }, [isLoading, summary, recentAlerts]);
  
  // Update "time ago" display every second
  useEffect(() => {
    const updateTimeAgo = () => {
      const now = new Date();
      const diffSeconds = Math.floor((now.getTime() - lastRefreshTime.getTime()) / 1000);
      
      if (diffSeconds < 5) {
        setTimeSinceRefresh(t('justNow'));
      } else if (diffSeconds < 60) {
        setTimeSinceRefresh(t('secondsAgo', diffSeconds));
      } else if (diffSeconds < 3600) {
        const mins = Math.floor(diffSeconds / 60);
        setTimeSinceRefresh(t('minutesAgo', mins));
      } else {
        const hours = Math.floor(diffSeconds / 3600);
        setTimeSinceRefresh(t('hoursAgo', hours));
      }
    };
    
    updateTimeAgo();
    const interval = setInterval(updateTimeAgo, 1000);
    return () => clearInterval(interval);
  }, [lastRefreshTime, lang]);

  // ── Computed KPI values ──────────────────────────────────────────────────
  const securityScore = calcSecurityScore(summary);
  const onlinePct = summary && summary.totalDevices > 0
    ? Math.round((summary.onlineDevices / summary.totalDevices) * 100)
    : 0;
  const avgConfidence = recentAlerts.length > 0
    ? Math.round(recentAlerts.reduce((s, a) => s + a.confidenceVal, 0) / recentAlerts.length * 100)
    : 0;

  // ── Chart data: Traffic Series (fallback to zeros if empty) ───────────────
  const chartData = trafficSeries.length > 0 ? trafficSeries : Array.from({ length: 24 }, (_, h) => {
    const pt: Record<string, any> = { hour: `${String(h).padStart(2, '0')}:00` };
    homes.forEach(home => { pt[home.code] = 0; });
    return pt;
  });

  return (
    <div className="space-y-6 font-sans">

      {/* ── Top Row: Status Bar & Refresh Controls ────────────────────────────────────────────── */}
      <div className="flex items-center justify-between mb-2 bg-slate-900/50 border border-slate-800 rounded-lg px-4 py-2">
        <div className="flex items-center gap-4">
          {/* Realtime Connection Status */}
          <div className="flex items-center gap-2">
            <div className="relative">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <div className="absolute inset-0 w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
            </div>
            <span className="text-xs font-bold text-emerald-400 uppercase tracking-widest">
              {t('live')}
            </span>
          </div>
          
          <div className="h-4 w-px bg-slate-700" />
          
          {/* Last Refresh Time */}
          <div className="flex items-center gap-2">
            <Clock size={12} className="text-slate-500" />
            <span className="text-xs font-medium text-slate-400">
              {t('updated')} <span className="font-bold text-slate-300">{timeSinceRefresh}</span>
            </span>
          </div>
          
          {/* Auto-refresh indicator */}
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <div className="w-1 h-1 rounded-full bg-blue-500 animate-pulse" />
            <span className="font-medium">{t('autoRefresh')}</span>
          </div>
        </div>
        
        {isAdmin && (
          <button
            onClick={onRefresh}
            disabled={isLoading}
            className="flex items-center gap-2 px-3 py-1.5 text-xs font-bold text-slate-400 hover:text-blue-400 bg-slate-900 border border-slate-800 hover:border-blue-500/30 rounded-lg transition-all cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <RefreshCw size={13} className={isLoading ? 'animate-spin' : ''} />
            {isLoading ? t('refreshing') : t('refreshNow')}
          </button>
        )}
      </div>

      {isAdmin ? (
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6">
          <KPICard
            icon={Server}
            title={t('totalDevices')}
            value={isLoading ? '—' : summary?.totalDevices ?? 0}
            subValue={t('onlinePct', onlinePct)}
            colorClass="text-emerald-400"
            loading={isLoading}
          />
          <KPICard
            icon={AlertTriangle}
            title={t('activeAlerts')}
            value={isLoading ? '—' : summary?.alertsToday ?? 0}
            subValue={t('criticalAlerts', summary?.criticalAlerts ?? 0)}
            colorClass="text-rose-400"
            loading={isLoading}
          />
          <KPICard
            icon={CloudLightning}
            title={t('cloudAiStatus')}
            value={modelStats ? `v${modelStats.version}` : '—'}
            subValue={modelStats ? modelStats.status.toUpperCase() : t('loading')}
            colorClass="text-blue-400"
            loading={isLoading}
          />
          <KPICard
            icon={TrendingUp}
            title={t('avgConfidence')}
            value={isLoading ? '—' : `${avgConfidence}%`}
            subValue={recentAlerts.length > 0 ? t('last20Alerts') : t('noData')}
            colorClass="text-amber-400"
            loading={isLoading}
          />
          <KPICard
            icon={ShieldCheck}
            title={t('securityScore')}
            value={isLoading ? '—' : securityScore}
            subValue={securityScore >= 90 ? t('excellent') : securityScore >= 70 ? t('good') : t('atRisk')}
            colorClass={securityScore >= 90 ? 'text-emerald-400' : securityScore >= 70 ? 'text-amber-400' : 'text-rose-400'}
            loading={isLoading}
          />
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <KPICard
            icon={Server}
            title={t('devices')}
            value={isLoading ? '—' : summary?.totalDevices ?? 0}
            subValue={t('onlineDevicesPct', onlinePct)}
            colorClass="text-emerald-400"
            loading={isLoading}
          />
          <KPICard
            icon={ShieldCheck}
            title={t('safetyScore')}
            value={isLoading ? '—' : securityScore}
            subValue={securityScore >= 90 ? t('safetyVeryGood') : securityScore >= 70 ? t('safetyGood') : t('safetyAttention')}
            colorClass={securityScore >= 90 ? 'text-emerald-400' : securityScore >= 70 ? 'text-amber-400' : 'text-rose-400'}
            loading={isLoading}
          />
        </div>
      )}


      {/* ── Middle Section: Traffic Chart + Alert Feed ─────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-10 gap-6 lg:h-[700px]">

        {/* Network Flow Monitor OR Forensic View (70%) */}
        <div className="lg:col-span-7 h-[600px] lg:h-full">
          <AnimatePresence mode="wait">
            {selectedAlert ? (
              <ForensicView
                key="forensic"
                alert={selectedAlert}
                onClose={() => setSelectedAlertId(null)}
                isAdmin={isAdmin}
                lang={lang}
                enableNode={enableNode}
                dismissAlert={dismissAlert}
              />
            ) : (
              <motion.div
                key="monitor"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl h-full flex flex-col"
              >
                <div className="flex justify-between items-center mb-6">
                  <div>
                    <h3 className="text-lg font-bold text-slate-100 tracking-tight">
                      {isAdmin ? t('flowTitleAdmin') : t('flowTitleUser')}
                    </h3>
                    <p className="text-[11px] text-slate-500 uppercase tracking-widest font-bold mt-2 border-l-2 border-slate-700 pl-2">
                      {isAdmin ? t('flowSubAdmin') : t('flowSubUser')}
                    </p>
                  </div>
                  <div className="flex items-center gap-4 flex-wrap max-w-sm justify-end">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-blue-500" />
                      <span className="text-xs font-bold text-slate-400 uppercase">
                        {t('flowsLabel')}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-rose-500" />
                      <span className="text-xs font-bold text-slate-400 uppercase">
                        {t('anomaliesLabel')}
                      </span>
                    </div>
                  </div>
                </div>

                {!networkFlowMetrics || networkFlowMetrics.length === 0 ? (
                  <div className="flex-1 flex flex-col items-center justify-center text-slate-600">
                    <Activity size={40} className="mb-3 opacity-30" />
                    <p className="text-sm font-semibold">{t('noFlowData')}</p>
                    <p className="text-xs text-slate-500 mt-2">{isAdmin ? t('waitingFlowData') : t('waitingDeviceData')}</p>
                    {isAdmin && <p className="text-xs text-slate-400 mt-1 font-mono">Debug: metrics = {JSON.stringify(networkFlowMetrics?.length || 0)}</p>}
                  </div>
                ) : (
                  <div className="flex-1 w-full min-h-0">
                    {isAdmin ? (
                      <div className="mb-4 text-xs text-slate-400 font-mono">
                        {t('flowSummaryAdmin', 
                          networkFlowMetrics.reduce((sum, m) => sum + (m.flows || 0), 0),
                          networkFlowMetrics.reduce((sum, m) => sum + (m.anomalies || 0), 0)
                        )}
                      </div>
                    ) : (
                      <div className="mb-4 text-xs text-slate-400 font-medium">
                        {t('flowSummaryUser', 
                          networkFlowMetrics.reduce((sum, m) => sum + (m.flows || 0), 0),
                          networkFlowMetrics.reduce((sum, m) => sum + (m.anomalies || 0), 0)
                        )}
                      </div>
                    )}
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={networkFlowMetrics}>
                        <defs>
                          <linearGradient id="colorFlows" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                          </linearGradient>
                          <linearGradient id="colorAnomalies" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                        <XAxis dataKey="hour" stroke="#64748b" fontSize={10} tickLine={false} />
                        <YAxis stroke="#64748b" fontSize={10} tickLine={false} axisLine={false} allowDecimals={false} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', color: '#f8fafc' }}
                          formatter={(value: number, name: string) => {
                            if (name === 'flows') return [value, t('flowsLabel')];
                            if (name === 'anomalies') return [value, t('anomaliesLabel')];
                            if (name === 'bytes') return [`${value} MB`, t('dataVolumeLabel')];
                            return [value, name];
                          }}
                        />
                        <Area
                          type="monotone"
                          dataKey="flows"
                          stroke="#3b82f6"
                          strokeWidth={2}
                          fillOpacity={1}
                          fill="url(#colorFlows)"
                        />
                        <Area
                          type="monotone"
                          dataKey="anomalies"
                          stroke="#ef4444"
                          strokeWidth={2}
                          fillOpacity={1}
                          fill="url(#colorAnomalies)"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Live Alert Feed (30%) */}
        <div className="lg:col-span-3 bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl flex flex-col h-[500px] lg:h-full">
          <div className="flex justify-between items-center mb-6 border-b border-slate-800 pb-4">
            <div className="flex items-center gap-3">
              <h3 className="text-lg font-bold text-slate-100 tracking-tight">{isAdmin ? t('liveAlertFeed') : t('latestAlerts')}</h3>
              <div className={`w-2.5 h-2.5 rounded-full shadow-lg ${isLoading ? 'bg-amber-500 animate-pulse shadow-amber-500/50' : 'bg-rose-500 animate-pulse shadow-rose-500/50'}`} />
            </div>
            {isAdmin && (
              <span className="text-[9px] bg-slate-800 text-slate-300 px-2 py-1 rounded font-black uppercase tracking-widest border border-slate-700">
                LIGHTGBM EDGE
              </span>
            )}
          </div>

          <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
            {recentAlerts.length === 0 && !isLoading && (
              <div className="flex flex-col items-center justify-center h-full text-slate-600 gap-2">
                <CheckCircle2 size={32} className="opacity-30" />
                <p className="text-sm font-semibold">{t('noAlertsFound')}</p>
              </div>
            )}
            {recentAlerts.map(alert => (
              <AlertItem
                key={alert.id}
                alert={alert}
                onClick={() => setSelectedAlertId(alert.id === selectedAlertId ? null : alert.id)}
                isSelected={alert.id === selectedAlertId}
                isAdmin={isAdmin}
                lang={lang}
              />
            ))}
          </div>
          <button className="w-full mt-6 py-3 text-xs font-bold text-slate-500 hover:text-slate-300 transition-colors border-t border-slate-800 cursor-pointer">
            {isAdmin ? t('viewAllIncidents') : t('viewAllAlerts')}
          </button>
        </div>
      </div>

      {/* ── Bottom Row: Only show for Admin ───────────────── */}
      {isAdmin && (
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Attack Type Distribution */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 shadow-xl">
          <div className="flex items-center gap-3 mb-8 border-b border-slate-800 pb-4">
            <Activity size={20} className="text-blue-400" />
            <h3 className="text-lg font-bold text-slate-100 tracking-tight">{t('attackDistTitle')}</h3>
            <span className="text-[10px] bg-slate-800 border border-slate-700 text-slate-400 rounded px-2 py-1 ml-auto font-bold uppercase tracking-widest">{t('last24h')}</span>
          </div>

          {attackDist.length === 0 ? (
            <div className="h-[280px] flex flex-col items-center justify-center text-slate-600 gap-2">
              <Activity size={28} className="opacity-30" />
              <p className="text-xs font-semibold">{t('noAttackData')}</p>
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <p className="text-3xl font-bold text-slate-200 tracking-tight">
                    {attackDist.reduce((a, b) => a + b.value, 0)}
                  </p>
                  <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mt-1">{t('totalRecorded')}</p>
                </div>
                <div className="text-right">
                  <p className="text-xl font-bold text-rose-400 tracking-tight">
                    {attackDist.length > 0 ? attackDist.reduce((max, obj) => obj.value > max.value ? obj : max).name : '—'}
                  </p>
                  <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mt-1">{t('topVector')}</p>
                </div>
              </div>
              <div className="h-[240px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={attackDist}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="45%"
                      innerRadius={60}
                      outerRadius={85}
                      stroke="none"
                      paddingAngle={2}
                    >
                      {attackDist.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} fillOpacity={0.9} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px', color: '#f8fafc' }}
                      itemStyle={{ color: '#fff', fontSize: '12px', fontWeight: 'bold' }}
                      formatter={(v: number) => [v, 'Count']}
                    />
                    <Legend
                      layout="horizontal"
                      verticalAlign="bottom"
                      align="center"
                      iconType="circle"
                      iconSize={8}
                      wrapperStyle={{ fontSize: '12px', fontWeight: 'bold', color: '#e2e8f0', paddingTop: '10px' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
        </div>

        {/* Edge Node Fleet (Admin only) */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 shadow-xl">
          <div className="flex items-center gap-3 mb-8 border-b border-slate-800 pb-4">
            <Server size={20} className="text-blue-400" />
            <h3 className="text-lg font-bold text-slate-100 tracking-tight">{t('edgeNodeFleet')}</h3>
            <span className="text-[10px] bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 rounded px-2 py-1 ml-auto font-bold uppercase tracking-widest">
              {t('nodesOnline', edgeNodes.filter(n => n.status === 'online').length, edgeNodes.length)}
            </span>
          </div>

          <div className="space-y-4 overflow-y-auto custom-scrollbar max-h-[320px] pr-2">
            {edgeNodes.length === 0 && !isLoading && (
              <p className="text-xs text-slate-600 text-center py-4">{t('noEdgeNodes')}</p>
            )}
            {edgeNodes.map(node => (
              <div key={node.id} className="flex flex-col gap-4 p-4 bg-slate-950/80 rounded-xl hover:bg-slate-900 border border-slate-800/80 transition-colors">
                <div className="flex items-center justify-between min-w-0">
                  <div className="flex items-center gap-3 min-w-0">
                    {node.status === 'online'
                      ? <Wifi size={16} className="text-emerald-500 flex-shrink-0" />
                      : <WifiOff size={16} className="text-rose-500 flex-shrink-0" />
                    }
                    <div className="min-w-0 flex items-center gap-3">
                      <p className="text-sm font-bold text-slate-100 truncate">{node.node_code}</p>
                      <span className="text-[10px] font-bold text-blue-400 bg-blue-500/10 border border-blue-500/20 px-2 py-0.5 rounded-md font-mono truncate">{node.location_text ?? node.ip_address ?? '—'}</span>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-2">
                  <div className="flex flex-col items-center justify-center bg-slate-900/50 py-2 rounded-lg border border-slate-800/50">
                    <div className="flex items-center gap-1.5 mb-1.5">
                      <Cpu size={12} className="text-slate-500" />
                      <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">{t('cpu')}</span>
                    </div>
                    <span className={`text-sm font-black font-mono ${(node.current_cpu_percent ?? 0) > 80 ? 'text-rose-400' : 'text-slate-200'}`}>
                      {node.current_cpu_percent?.toFixed(0) ?? '—'}%
                    </span>
                  </div>
                  <div className="flex flex-col items-center justify-center bg-slate-900/50 py-2 rounded-lg border border-slate-800/50">
                    <div className="flex items-center gap-1.5 mb-1.5">
                      <Database size={12} className="text-slate-500" />
                      <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">{t('ram')}</span>
                    </div>
                    <span className={`text-sm font-black font-mono ${(node.current_ram_percent ?? 0) > 80 ? 'text-rose-400' : 'text-slate-200'}`}>
                      {node.current_ram_percent?.toFixed(0) ?? '—'}%
                    </span>
                  </div>
                  <div className="flex flex-col items-center justify-center bg-slate-900/50 py-2 rounded-lg border border-slate-800/50">
                    <div className="flex items-center gap-1.5 mb-1.5">
                      <Thermometer size={12} className="text-slate-500" />
                      <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">{t('temp')}</span>
                    </div>
                    <span className={`text-sm font-black font-mono ${(node.current_temperature_c ?? 0) > 70 ? 'text-rose-400' : 'text-slate-200'}`}>
                      {node.current_temperature_c?.toFixed(0) ?? '—'}°
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* AI Performance Panel */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 shadow-xl">
          <div className="flex items-center gap-3 mb-8 border-b border-slate-800 pb-4">
            <Brain size={20} className="text-blue-400" />
            <h3 className="text-lg font-bold text-slate-100 tracking-tight">{t('aiPerfPanel')}</h3>
          </div>

          {modelStats ? (
            <>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <p className="text-3xl font-bold text-slate-200 tracking-tight">
                    {modelStats.f1_score != null ? `${(modelStats.f1_score * 100).toFixed(1)}%` : '—'}
                  </p>
                  <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mt-1">{t('f1Score')}</p>
                </div>
                <div className="text-right">
                  <p className="text-lg font-bold text-emerald-400 tracking-tight">
                    {modelStats.accuracy != null ? `${(modelStats.accuracy * 100).toFixed(1)}%` : '—'}
                  </p>
                  <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mt-1">{t('accuracy')}</p>
                </div>
              </div>

              {/* Model meta */}
              <div className="space-y-1 mb-6 mt-6">
                {[
                  { label: t('precision'), val: modelStats.precision, suffix: '%', mult: 100 },
                  { label: t('recall'), val: modelStats.recall, suffix: '%', mult: 100 },
                  { label: t('fpRate'), val: modelStats.false_positive_rate, suffix: '%', mult: 100 },
                  { label: t('latency'), val: modelStats.latency_ms, suffix: 'ms', mult: 1 },
                  { label: t('throughput'), val: modelStats.throughput_per_s, suffix: '/s', mult: 1 },
                ].map(({ label, val, suffix, mult }) => (
                  <div key={label} className="flex justify-between items-center text-sm py-2 px-3 rounded-lg hover:bg-slate-800/30 transition-colors border border-transparent hover:border-slate-800/50">
                    <span className="text-slate-400 font-semibold">{label}</span>
                    <span className="text-slate-100 font-mono font-black">
                      {val != null ? `${(val * mult).toFixed(mult === 100 ? 1 : 0)}${suffix}` : '—'}
                    </span>
                  </div>
                ))}
              </div>

              <div className="flex items-center justify-between pt-4 border-t border-slate-800">
                <span className="text-[11px] text-slate-500 font-mono">{modelStats.version}</span>
                <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded border ${modelStats.status === 'deployed'
                    ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                    : 'bg-blue-500/15 text-blue-400 border-blue-500/30'
                  }`}>{modelStats.status}</span>
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-[200px] text-slate-600 gap-2">
              <Brain size={32} className="opacity-30" />
              <p className="text-xs font-semibold">{t('noModelData')}</p>
            </div>
          )}
        </div>
      </div>
      )}

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #334155; }
      `}</style>
    </div>
  );
};
