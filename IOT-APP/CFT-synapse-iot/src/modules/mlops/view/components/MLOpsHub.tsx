import React, { useState, useEffect } from 'react';
import { 
  Radar, 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell
} from 'recharts';
import ReactECharts from 'echarts-for-react';
import { 
  Activity, 
  History, 
  RefreshCw, 
  Settings, 
  ShieldCheck, 
  Zap, 
  CheckCircle2, 
  ArrowUpRight,
  ArrowDownRight,
  Loader2,
  Terminal,
  Wifi,
  WifiOff,
  X,
  Rocket,
  Trash2,
} from 'lucide-react';
import { useMLOpsData } from '../../hooks/useMLOpsData';
import type { VersionLog } from '../../hooks/useMLOpsData';
import { useDbStatus } from '../../../../core/useDbStatus';
import { supabase } from '../../../../core/lib/supabaseClient';
import { useMlops } from '../../controller/useMlops';
import { useAuthContext } from '../../../../core/auth/AuthProvider';
import { useLanguage } from '../../../../core/i18n/LanguageContext';

// --- Sub-components ---

const FidelityGauge = ({ score }: { score: number }) => {
  const option = {
    series: [
      {
        type: 'gauge',
        startAngle: 180,
        endAngle: 0,
        min: 0,
        max: 100,
        splitNumber: 5,
        axisLine: {
          lineStyle: {
            width: 6,
            color: [
              [0.3, '#ef4444'],
              [0.7, '#f59e0b'],
              [1, '#10b981']
            ]
          }
        },
        pointer: {
          icon: 'path://M12.8,0.7l12,10.1c0.4,0.3,0.4,0.9,0.1,1.2c-0.3,0.4-0.9,0.4-1.2,0.1L12,2.3L0.3,12.1c-0.4,0.3-1,0.3-1.3-0.1c-0.3-0.4-0.3-1,0.1-1.3l12-10.1C11.5,0.3,12.2,0.3,12.8,0.7z',
          length: '12%',
          width: 20,
          offsetCenter: [0, '-60%'],
          itemStyle: { color: 'auto' }
        },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        title: { show: false },
        detail: {
          valueAnimation: true,
          formatter: '{value}%',
          color: '#f8fafc',
          fontSize: 20,
          offsetCenter: [0, '20%']
        },
        data: [{ value: score }]
      }
    ]
  };

  return <ReactECharts option={option} style={{ height: '180px' }} />;
};

const DbBadge = () => {
  const { connected, loading } = useDbStatus();
  if (loading) return (
    <div className="flex items-center gap-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
      <Loader2 size={12} className="animate-spin" /> Connecting…
    </div>
  );
  return connected ? (
    <div className="flex items-center gap-2 text-[10px] font-bold text-emerald-400 uppercase tracking-widest">
      <Wifi size={12} /> DB Connected
    </div>
  ) : (
    <div className="flex items-center gap-2 text-[10px] font-bold text-rose-400 uppercase tracking-widest">
      <WifiOff size={12} /> DB Offline
    </div>
  );
};

// Metric comparison rows derived from top 2 versions
function buildCompareRows(versionHistory: ReturnType<typeof useMLOpsData>['versionHistory']) {
  const v_new = versionHistory[0];
  const v_old = versionHistory[1];
  if (!v_new) return [];
  const fmt = (n: number | null | undefined, unit: string) =>
    n != null ? `${(n * (unit === '%' ? 100 : 1)).toFixed(unit === 'ms' ? 0 : 1)}${unit}` : '—';
  return [
    { label: 'F1-Score', old: fmt(v_old?.f1_score, '%'), new: fmt(v_new.f1_score, '%'), trend: (v_new.f1_score ?? 0) > (v_old?.f1_score ?? 0) ? 'up' : 'down' },
    { label: 'False Positive Rate', old: fmt(v_old?.false_positive_rate, '%'), new: fmt(v_new.false_positive_rate, '%'), trend: (v_new.false_positive_rate ?? 0) < (v_old?.false_positive_rate ?? 1) ? 'down' : 'up' },
    { label: 'Inference Latency', old: fmt(v_old?.latency_ms ?? null, 'ms'), new: fmt(v_new.latency_ms, 'ms'), trend: (v_new.latency_ms ?? 9999) < (v_old?.latency_ms ?? 9999) ? 'down' : 'up' },
    { label: 'Memory Footprint', old: v_old?.memory_mb != null ? `${v_old.memory_mb}MB` : '—', new: v_new.memory_mb != null ? `${v_new.memory_mb}MB` : '—', trend: (v_new.memory_mb ?? 9999) < (v_old?.memory_mb ?? 9999) ? 'down' : 'up' },
    { label: 'Throughput', old: v_old?.throughput_per_s != null ? `${v_old.throughput_per_s.toFixed(0)}/s` : '—', new: v_new.throughput_per_s != null ? `${v_new.throughput_per_s.toFixed(0)}/s` : '—', trend: (v_new.throughput_per_s ?? 0) > (v_old?.throughput_per_s ?? 0) ? 'up' : 'down' },
  ];
}

function isProductionLike(status: string): boolean {
  return ['production', 'deployed', 'active'].includes((status || '').toLowerCase());
}

export const MLOpsHub: React.FC = () => {
  const [isRetraining, setIsRetraining] = useState(false);
  const [isDeploying, setIsDeploying] = useState(false);
  const [isClearingHistory, setIsClearingHistory] = useState(false);
  const [isClearingDeployments, setIsClearingDeployments] = useState(false);
  const [deletingVersionId, setDeletingVersionId] = useState<string | null>(null);
  const [batchSize, setBatchSize] = useState(128);
  const [learningRate, setLearningRate] = useState(0.001);
  const [detailVersion, setDetailVersion] = useState<VersionLog | null>(null);
  const [promotingId, setPromotingId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  
  // Security: Rate limiting - track last action timestamps
  const [lastDeployTime, setLastDeployTime] = useState<number>(0);
  const [lastRetrainTime, setLastRetrainTime] = useState<number>(0);
  const DEPLOY_COOLDOWN_MS = 30000; // 30 seconds between deploys
  const RETRAIN_COOLDOWN_MS = 60000; // 60 seconds between retrains

  const {
    versionHistory,
    radarData,
    benchmarkData,
    pipelineSteps,
    activeVersion,
    fidelityScore,
    fidelityIsProxy,
    activeJob,
    triggerRetrain,
    promoteVersionToProduction,
    clearHistory,
    loading,
    error,
  } = useMLOpsData();

  // Get deployModel and clearDeploymentHistory from controller
  const mlopsController = useMlops();
  
  const { t } = useLanguage();

  // Get user role for authorization
  const { role } = useAuthContext();
  
  // Security: Role-based access control
  // SOC level 3 is treated as admin (highest privilege)
  // If role is null (not set in database), default to admin for backward compatibility
  const isAdmin = role === 'admin' || role === 'soc level 3' || role === null;
  const isOperator = role === 'operator';
  
  const canDeploy = isAdmin; // Only admin and SOC level 3 can deploy
  const canRetrain = isAdmin || isOperator; // Admin, SOC level 3, and operator can retrain
  
  // Debug: Log role for troubleshooting
  useEffect(() => {
    console.log('[MLOps Security] Current user role:', role);
    console.log('[MLOps Security] Is admin:', isAdmin);
    console.log('[MLOps Security] Can deploy:', canDeploy);
    console.log('[MLOps Security] Can retrain:', canRetrain);
  }, [role, isAdmin, canDeploy, canRetrain]);

  const handleRetrain = async () => {
    // Security: Check authorization
    if (!canRetrain) {
      setActionError(`Access denied: Your role "${role}" cannot trigger retraining. Required: admin, SOC level 3, or operator`);
      return;
    }
    
    // Security: Rate limiting
    const now = Date.now();
    const timeSinceLastRetrain = now - lastRetrainTime;
    if (timeSinceLastRetrain < RETRAIN_COOLDOWN_MS) {
      const remainingSeconds = Math.ceil((RETRAIN_COOLDOWN_MS - timeSinceLastRetrain) / 1000);
      setActionError(`Rate limit: Please wait ${remainingSeconds} seconds before triggering another retrain`);
      return;
    }
    
    setIsRetraining(true);
    setActionError(null);
    try {
      // Use the hook's triggerRetrain (calls Lambda directly)
      await triggerRetrain({ batchSize, learningRate });
      setLastRetrainTime(now);
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Retrain failed');
    } finally {
      setIsRetraining(false);
    }
  };

  const handleDeployToEdge = async (versionId: string) => {
    // Security: Check authorization
    if (!canDeploy) {
      setActionError(`Access denied: Your role "${role}" cannot deploy models. Required: admin or SOC level 3`);
      return;
    }
    
    // Security: Rate limiting
    const now = Date.now();
    const timeSinceLastDeploy = now - lastDeployTime;
    if (timeSinceLastDeploy < DEPLOY_COOLDOWN_MS) {
      const remainingSeconds = Math.ceil((DEPLOY_COOLDOWN_MS - timeSinceLastDeploy) / 1000);
      setActionError(`Rate limit: Please wait ${remainingSeconds} seconds before deploying again`);
      return;
    }
    
    if (!window.confirm('Deploy this model version to all edge nodes?')) return;
    setIsDeploying(true);
    setActionError(null);
    try {
      // Security: Validate model version exists and is production-ready
      const { data: modelVersion, error: modelError } = await supabase
        .from('model_versions')
        .select('id, version, status, artifact_uri')
        .eq('id', versionId)
        .single();
      
      if (modelError) throw new Error(`Failed to validate model: ${modelError.message}`);
      if (!modelVersion) throw new Error('Model version not found');
      if (!isProductionLike(modelVersion.status)) {
        throw new Error(`Model status is "${modelVersion.status}" - only production models can be deployed`);
      }
      if (!modelVersion.artifact_uri) {
        throw new Error('Model artifact URI is missing - cannot deploy');
      }
      
      // Fetch all online edge nodes from Supabase
      const { data: nodes, error } = await supabase
        .from('edge_nodes')
        .select('id')
        .eq('status', 'online');
      
      if (error) throw new Error(`Failed to fetch edge nodes: ${error.message}`);
      if (!nodes || nodes.length === 0) {
        setActionError('No online edge nodes found');
        return;
      }
      
      const nodeIds = nodes.map(n => n.id);
      
      // Use deployModel from controller
      await mlopsController.deployModel({ modelVersionId: versionId, targetNodeIds: nodeIds });
      
      setLastDeployTime(now);
      alert(`Successfully deployed model ${modelVersion.version} to ${nodeIds.length} edge node(s). Check Deployments table for status.`);
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Deploy failed');
    } finally {
      setIsDeploying(false);
    }
  };

  const handlePromote = async (v: VersionLog) => {
    // Security: Check authorization
    if (!canDeploy) {
      setActionError(`Access denied: Your role "${role}" cannot promote models. Required: admin or SOC level 3`);
      return;
    }
    
    if (!window.confirm(`Set ${v.version} as production for this model? Other versions of the same model will be marked archived.`)) return;
    setPromotingId(v.id);
    setActionError(null);
    const r = await promoteVersionToProduction(v.id, v.modelId);
    setPromotingId(null);
    if (!r.ok) setActionError(r.error ?? 'Promote failed');
  };

  const handleClearHistory = async () => {
    // Security: Check authorization
    if (!canDeploy) {
      setActionError(`Access denied: Your role "${role}" cannot clear history. Required: admin or SOC level 3`);
      return;
    }
    
    if (!window.confirm('Delete old model versions? (Keep only the latest 5 versions)\n\nThis action cannot be undone.')) return;
    setIsClearingHistory(true);
    setActionError(null);
    try {
      const r = await clearHistory();
      if (!r.ok) {
        setActionError(r.error ?? 'Clear history failed');
      } else {
        const count = (r as any).deletedCount ?? 0;
        alert(`Successfully deleted ${count} old model version(s)`);
      }
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Clear history failed');
    } finally {
      setIsClearingHistory(false);
    }
  };

  const handleClearDeployments = async () => {
    // Security: Check authorization
    if (!canDeploy) {
      setActionError(`Access denied: Your role "${role}" cannot clear deployment history. Required: admin or SOC level 3`);
      return;
    }
    
    if (!window.confirm('Delete all completed/failed/cancelled deployments? This action cannot be undone.')) return;
    setIsClearingDeployments(true);
    setActionError(null);
    try {
      const r = await mlopsController.clearDeploymentHistory();
      if (!r.ok) {
        setActionError(r.error ?? 'Clear deployment history failed');
      } else {
        alert('Deployment history cleared successfully');
      }
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Clear deployment history failed');
    } finally {
      setIsClearingDeployments(false);
    }
  };

  const handleDeleteVersion = async (versionId: string, versionName: string) => {
    // Security: Check authorization
    if (!canDeploy) {
      setActionError(`Access denied: Your role "${role}" cannot delete model versions. Required: admin or SOC level 3`);
      return;
    }
    
    if (!window.confirm(`Delete model version "${versionName}"?\n\nThis will:\n- Delete all related deployments\n- Delete all related flow inferences\n- Delete all related alerts\n- Remove this version from edge nodes\n- Remove this version from retrain jobs\n\nThis action cannot be undone.`)) return;
    setDeletingVersionId(versionId);
    setActionError(null);
    try {
      // Step 1: Update edge_nodes to set deployed_model_version_id to NULL
      const { error: edgeError } = await supabase
        .from('edge_nodes')
        .update({ deployed_model_version_id: null })
        .eq('deployed_model_version_id', versionId);
      
      if (edgeError) {
        setActionError(`Failed to update edge nodes: ${edgeError.message}`);
        setDeletingVersionId(null);
        return;
      }
      
      // Step 2: Delete all flow_inference records that reference this model version
      const { error: flowError } = await supabase
        .from('flow_inference' as any)
        .delete()
        .eq('model_version_id', versionId);
      
      if (flowError) {
        setActionError(`Failed to delete flow inferences: ${flowError.message}`);
        setDeletingVersionId(null);
        return;
      }
      
      // Step 3: Update retrain_jobs_all to set all version references to NULL
      const { error: retrainError1 } = await supabase
        .from('retrain_jobs_all' as any)
        .update({ 
          job_teacher_from_version_id: null,
          job_teacher_to_version_id: null,
          job_student_from_version_id: null,
          job_student_to_version_id: null
        })
        .or(`job_teacher_from_version_id.eq.${versionId},job_teacher_to_version_id.eq.${versionId},job_student_from_version_id.eq.${versionId},job_student_to_version_id.eq.${versionId}`);
      
      if (retrainError1) {
        setActionError(`Failed to update retrain jobs: ${retrainError1.message}`);
        setDeletingVersionId(null);
        return;
      }
      
      // Step 4: Update alerts_all to set model version references to NULL
      const { error: alertsError } = await supabase
        .from('alerts_all' as any)
        .update({ 
          alert_edge_model_version_id: null,
          alert_cloud_model_version_id: null
        })
        .or(`alert_edge_model_version_id.eq.${versionId},alert_cloud_model_version_id.eq.${versionId}`);
      
      if (alertsError) {
        setActionError(`Failed to update alerts: ${alertsError.message}`);
        setDeletingVersionId(null);
        return;
      }
      
      // Step 5: Delete all deployments that reference this model version
      const { error: deployError } = await supabase
        .from('deployments_all' as any)
        .delete()
        .eq('deployment_model_version_id', versionId);
      
      if (deployError) {
        setActionError(`Failed to delete deployments: ${deployError.message}`);
        setDeletingVersionId(null);
        return;
      }
      
      // Step 6: Finally, delete the model version
      const { error: versionError } = await supabase
        .from('model_versions')
        .delete()
        .eq('id', versionId);
      
      if (versionError) {
        setActionError(`Failed to delete version: ${versionError.message}`);
      } else {
        // Refresh data after successful deletion
        window.location.reload();
      }
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Delete version failed');
    } finally {
      setDeletingVersionId(null);
    }
  };

  const compareRows = buildCompareRows(versionHistory);
  const v_old_label = versionHistory[1]?.version ?? 'v_old';
  const v_new_label = versionHistory[0]?.version ?? 'v_new';

  return (
    <div className="space-y-6">
      {/* Header Section */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-100 flex items-center gap-3">
            <Settings className="text-emerald-500" /> {t('mlops', 'title')}
          </h2>
          <p className="text-slate-400 text-sm mt-1">{t('mlops', 'subTitle')}</p>
        </div>
        <div className="flex items-center gap-3">
          <DbBadge />
          <div className="bg-slate-900/50 px-4 py-2 rounded-lg border border-slate-800 flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-xs font-bold text-slate-300 uppercase tracking-widest">
              Active Model: {loading ? '…' : activeVersion}
            </span>
          </div>
          <button 
            onClick={handleClearHistory}
            disabled={isClearingHistory || !canDeploy}
            title={!canDeploy ? 'Only admin and SOC level 3 roles can clear history' : 'Clear archived model versions'}
            className="px-4 py-2 bg-rose-600 hover:bg-rose-500 disabled:bg-slate-800 disabled:cursor-not-allowed text-white text-xs font-bold uppercase tracking-widest rounded-lg transition-all flex items-center gap-2 shadow-lg shadow-rose-900/20"
          >
            {isClearingHistory ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
            {t('mlops', 'clearHistory') || 'Clear History'}
          </button>
          <button 
            onClick={handleClearDeployments}
            disabled={isClearingDeployments || !canDeploy}
            title={!canDeploy ? 'Only admin and SOC level 3 roles can clear deployment history' : 'Clear completed/failed deployments'}
            className="px-4 py-2 bg-orange-600 hover:bg-orange-500 disabled:bg-slate-800 disabled:cursor-not-allowed text-white text-xs font-bold uppercase tracking-widest rounded-lg transition-all flex items-center gap-2 shadow-lg shadow-orange-900/20"
          >
            {isClearingDeployments ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
            {t('mlops', 'clearDeployments') || 'Clear Deployments'}
          </button>
          <button 
            onClick={handleRetrain}
            disabled={isRetraining || !canRetrain}
            title={!canRetrain ? 'Only admin, SOC level 3, and operator roles can trigger retraining' : 'Trigger model retraining'}
            className="px-6 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:cursor-not-allowed text-white text-xs font-bold uppercase tracking-widest rounded-lg transition-all flex items-center gap-2 shadow-lg shadow-emerald-900/20"
          >
            {isRetraining ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
            {t('mlops', 'triggerTraining')}
          </button>
        </div>
      </div>

      {(error || actionError) && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-lg px-4 py-3 text-xs text-rose-400 font-mono">
          ⚠ {error || actionError}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Top Row: Model Comparison Audit (8/12) */}
        <div className="lg:col-span-8 bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-xl p-6 shadow-xl">
          <div className="flex justify-between items-center mb-8">
            <h3 className="text-sm font-bold text-slate-200 uppercase tracking-widest flex items-center gap-2">
              <ShieldCheck size={16} className="text-blue-400" /> {t('mlops', 'modelCompare') || 'Model Comparison Audit'}
            </h3>
            <div className="flex gap-4">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-slate-700" />
                <span className="text-[10px] font-bold text-slate-500 uppercase">{v_old_label} (Old)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-blue-500" />
                <span className="text-[10px] font-bold text-slate-500 uppercase">{v_new_label} (Current)</span>
              </div>
            </div>
          </div>

          {loading ? (
            <div className="flex items-center justify-center h-48 text-slate-500 text-xs">
              <Loader2 size={20} className="animate-spin mr-2" /> Loading model data…
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Radar Chart */}
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                    <PolarGrid stroke="#1e293b" />
                    <PolarAngleAxis dataKey="subject" tick={{ fill: '#64748b', fontSize: 10 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                    <Radar name={v_old_label} dataKey="v_old" stroke="#475569" fill="#475569" fillOpacity={0.3} />
                    <Radar name={v_new_label} dataKey="v_new" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.5} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '10px' }} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              {/* Comparison Table */}
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest pb-2 border-b border-slate-800">
                  <span>Metric</span>
                  <span>{v_old_label}</span>
                  <span>{v_new_label}</span>
                </div>
                <div className="space-y-3">
                  {compareRows.map((m, i) => (
                    <div key={i} className="grid grid-cols-3 gap-4 items-center">
                      <span className="text-xs text-slate-400">{m.label}</span>
                      <span className="text-xs font-mono text-slate-500">{m.old}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono text-slate-200">{m.new}</span>
                        {m.trend === 'up' ? <ArrowUpRight size={12} className="text-emerald-500" /> : <ArrowDownRight size={12} className="text-emerald-500" />}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Top Row: Distillation Performance (4/12) */}
        <div className="lg:col-span-4 bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-xl p-6 shadow-xl flex flex-col">
          <h3 className="text-sm font-bold text-slate-200 uppercase tracking-widest flex items-center gap-2 mb-8">
            <Zap size={16} className="text-amber-400" /> {t('mlops', 'distillationMetrics') || 'Distillation Metrics'}
          </h3>

          <div className="flex-1 space-y-8">
            <div className="text-center">
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2 block">Knowledge Fidelity Score</span>
              <FidelityGauge score={fidelityScore} />
              <p className="text-[10px] text-slate-500 mt-2 italic px-4">
                {fidelityIsProxy
                  ? 'Proxy: derived from latest version F1. Add knowledge_fidelity / distillation_fidelity to model_versions.metrics_json (pipeline export) for a true Student↔Teacher fidelity score.'
                  : 'Distillation fidelity from metrics_json (Teacher vs Student agreement).'}
              </p>
            </div>

            <div className="space-y-4">
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block">Edge vs. Cloud Accuracy</span>
              <div className="h-[150px]">
                {loading ? (
                  <div className="flex items-center justify-center h-full text-slate-600 text-xs">Loading…</div>
                ) : (
                  <div className="w-full h-full min-h-[150px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={benchmarkData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                      <XAxis type="number" domain={[0, 100]} hide />
                      <YAxis dataKey="name" type="category" fontSize={10} tick={{ fill: '#64748b' }} width={80} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '10px' }} />
                      <Bar dataKey="accuracy" radius={[0, 4, 4, 0]} barSize={20}>
                        {benchmarkData.map((_, index) => (
                          <Cell key={`cell-${index}`} fill={index === 0 ? '#f59e0b' : '#3b82f6'} />
                        ))}
                      </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Row: Pipeline Control (4/12) */}
        <div className="lg:col-span-4 bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-xl p-6 shadow-xl">
          <h3 className="text-sm font-bold text-slate-200 uppercase tracking-widest flex items-center gap-2 mb-4">
            <Activity size={16} className="text-emerald-400" /> {t('mlops', 'automatedPipeline') || 'Automated Pipeline'}
          </h3>

          {activeJob.jobId && (activeJob.jobStatus === 'queued' || activeJob.jobStatus === 'running') && (
            <div className="mb-6 rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-3 py-3 space-y-2">
              <div className="flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-400">
                <span>Job {activeJob.jobId.slice(0, 8)}… ({activeJob.jobStatus})</span>
                <span className="text-emerald-400">{activeJob.progressPercent}%</span>
              </div>
              <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-emerald-500 transition-all duration-500"
                  style={{ width: `${activeJob.progressPercent}%` }}
                />
              </div>
              {activeJob.eventStep && (
                <p className="text-[10px] text-slate-500 font-mono">Step: {activeJob.eventStep}</p>
              )}
              {activeJob.eventMessage && (
                <p className="text-[10px] text-slate-400 font-mono leading-snug">{activeJob.eventMessage}</p>
              )}
            </div>
          )}

          <div className="space-y-6">
            <div className="relative pl-8 space-y-8">
              <div className="absolute left-3.5 top-2 bottom-2 w-px bg-slate-800" />
              {pipelineSteps.map((step) => (
                <div key={step.id} className="relative flex items-center gap-4">
                  <div className={`absolute -left-[26px] w-5 h-5 rounded-full border-2 flex items-center justify-center z-10 ${
                    step.status === 'completed' ? 'bg-emerald-500 border-emerald-500 text-slate-900' :
                    step.status === 'active' ? 'bg-slate-900 border-emerald-500 text-emerald-500 animate-pulse' :
                    'bg-slate-900 border-slate-800 text-slate-700'
                  }`}>
                    {step.status === 'completed' ? <CheckCircle2 size={12} /> : step.id}
                  </div>
                  <div className="flex-1">
                    <p className={`text-xs font-bold uppercase tracking-tight ${
                      step.status === 'completed' ? 'text-slate-200' :
                      step.status === 'active' ? 'text-emerald-400' : 'text-slate-600'
                    }`}>{step.name}</p>
                    {step.status === 'active' && (
                      <div className="mt-2 h-1 w-full bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-emerald-500 transition-all duration-500"
                          style={{ width: `${activeJob.progressPercent}%` }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            <div className="bg-slate-950/50 rounded-lg border border-slate-800 p-4 space-y-4">
              <div className="flex items-center gap-2 text-[10px] font-bold text-slate-500 uppercase">
                <Terminal size={12} /> Training Configuration
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-[8px] text-slate-600 uppercase font-bold">Batch Size</label>
                  <input 
                    type="number" 
                    value={batchSize}
                    onChange={(e) => setBatchSize(parseInt(e.target.value) || 0)}
                    className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs font-mono text-slate-300 focus:outline-none focus:border-emerald-500" 
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-[8px] text-slate-600 uppercase font-bold">Learning Rate</label>
                  <input 
                    type="text" 
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0)}
                    className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs font-mono text-slate-300 focus:outline-none focus:border-emerald-500" 
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Row: Version History (8/12) */}
        <div className="lg:col-span-8 bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-xl p-6 shadow-xl flex flex-col">
          <h3 className="text-sm font-bold text-slate-200 uppercase tracking-widest flex items-center gap-2 mb-8">
            <History size={16} className="text-slate-400" /> {t('mlops', 'modelRegistry')}
          </h3>

          <div className="flex-1 overflow-x-auto">
            {loading ? (
              <div className="flex items-center justify-center h-24 text-slate-500 text-xs">
                <Loader2 size={16} className="animate-spin mr-2" /> Loading versions…
              </div>
            ) : versionHistory.length === 0 ? (
              <p className="text-center text-slate-600 text-xs py-8">No model versions found in database.</p>
            ) : (
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-800">
                    <th className="pb-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">{t('mlops', 'colVersion') || 'Version'}</th>
                    <th className="pb-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">{t('mlops', 'colTimestamp') || 'Timestamp'}</th>
                    <th className="pb-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">{t('mlops', 'colAccuracy') || 'Accuracy'}</th>
                    <th className="pb-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">{t('mlops', 'colStatus') || 'Status'}</th>
                    <th className="pb-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">{t('mlops', 'colActions') || 'Actions'}</th>
                  </tr>
                </thead>
                <tbody className="text-xs font-mono">
                  {versionHistory.map((v) => (
                    <tr key={v.id} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                      <td className="py-4 text-slate-200 font-bold">{v.version}</td>
                      <td className="py-4 text-slate-500">{v.timestamp}</td>
                      <td className="py-4 text-slate-300">{(v.accuracy * 100).toFixed(1)}%</td>
                      <td className="py-4">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                          isProductionLike(v.status) ? 'bg-emerald-500/10 text-emerald-400' :
                          v.status === 'archived' ? 'bg-slate-800 text-slate-500' : 'bg-rose-500/10 text-rose-400'
                        }`}>
                          {v.status}
                        </span>
                      </td>
                      <td className="py-4">
                        <div className="flex gap-2">
                          <button
                            type="button"
                            title="Metrics detail"
                            onClick={() => setDetailVersion(v)}
                            className="p-1.5 hover:bg-slate-800 rounded transition-colors text-slate-400 hover:text-slate-200"
                          >
                            <History size={14} />
                          </button>
                          {isProductionLike(v.status) && (
                            <button
                              type="button"
                              title={!canDeploy ? 'Only admin and SOC level 3 roles can deploy models' : 'Deploy to Edge'}
                              disabled={isDeploying || !canDeploy}
                              onClick={() => handleDeployToEdge(v.id)}
                              className="p-1.5 hover:bg-blue-500/10 rounded transition-colors text-blue-500 hover:text-blue-400 disabled:opacity-40 disabled:cursor-not-allowed"
                            >
                              {isDeploying ? <Loader2 size={14} className="animate-spin" /> : <Rocket size={14} />}
                            </button>
                          )}
                          {!isProductionLike(v.status) && (
                            <button
                              type="button"
                              title={!canDeploy ? 'Only admin and SOC level 3 roles can promote models' : 'Promote to production'}
                              disabled={promotingId === v.id || !canDeploy}
                              onClick={() => handlePromote(v)}
                              className="p-1.5 hover:bg-emerald-500/10 rounded transition-colors text-emerald-500 hover:text-emerald-400 disabled:opacity-40 disabled:cursor-not-allowed"
                            >
                              {promotingId === v.id ? <Loader2 size={14} className="animate-spin" /> : <Rocket size={14} />}
                            </button>
                          )}
                          <button
                            type="button"
                            title={!canDeploy ? 'Only admin and SOC level 3 roles can delete model versions' : 'Delete this version'}
                            disabled={deletingVersionId === v.id || !canDeploy}
                            onClick={() => handleDeleteVersion(v.id, v.version)}
                            className="p-1.5 hover:bg-rose-500/10 rounded transition-colors text-rose-500 hover:text-rose-400 disabled:opacity-40 disabled:cursor-not-allowed"
                          >
                            {deletingVersionId === v.id ? <Loader2 size={14} className="animate-spin" /> : <X size={14} />}
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>

      {detailVersion && (
        <>
          <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-[9998]" onClick={() => setDetailVersion(null)} />
          <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4 pointer-events-none">
            <div
              className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-w-lg w-full max-h-[80vh] overflow-hidden pointer-events-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800">
                <div>
                  <h4 className="text-slate-100 font-bold text-sm">{detailVersion.version}</h4>
                  <p className="text-[10px] text-slate-500 font-mono">{detailVersion.id}</p>
                </div>
                <button type="button" onClick={() => setDetailVersion(null)} className="p-2 rounded-lg hover:bg-slate-800 text-slate-400">
                  <X size={18} />
                </button>
              </div>
              <div className="p-4 overflow-y-auto max-h-[60vh] text-xs space-y-3">
                <div className="grid grid-cols-2 gap-2 text-slate-400">
                  <span>Status</span><span className="text-slate-200">{detailVersion.status}</span>
                  <span>Author</span><span className="text-slate-200">{detailVersion.author}</span>
                  <span>F1</span><span className="text-slate-200">{(detailVersion.f1_score * 100).toFixed(2)}%</span>
                  <span>Latency</span><span className="text-slate-200">{detailVersion.latency_ms} ms</span>
                </div>
                <div>
                  <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">metrics_json</p>
                  <pre className="text-[10px] text-slate-300 bg-slate-950 border border-slate-800 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap font-mono">
                    {detailVersion.metricsJson
                      ? (() => {
                          try {
                            return JSON.stringify(JSON.parse(detailVersion.metricsJson), null, 2);
                          } catch {
                            return detailVersion.metricsJson;
                          }
                        })()
                      : '—'}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default MLOpsHub;
