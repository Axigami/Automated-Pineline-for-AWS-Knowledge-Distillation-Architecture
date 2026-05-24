import React from 'react';
import { 
  BrainCircuit, 
  Activity, 
  Zap, 
  AlertCircle, 
  Play, 
  Settings2, 
  CheckCircle2, 
  Loader2, 
  ArrowRight,
  TrendingUp,
  Target,
  ShieldCheck
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { motion } from 'motion/react';

const DRIFT_DATA = Array.from({ length: 20 }, (_, i) => ({
  time: i,
  drift: 0.1 + Math.sin(i / 3) * 0.05 + (i > 15 ? 0.2 : 0),
  baseline: 0.1,
}));

const CONFUSION_MATRIX = [
  { label: 'Benign', values: [98, 1, 0, 1, 0] },
  { label: 'Botnet', values: [2, 92, 4, 1, 1] },
  { label: 'DDoS', values: [0, 3, 95, 2, 0] },
  { label: 'DoS', values: [1, 2, 1, 94, 2] },
  { label: 'PortScan', values: [0, 1, 0, 1, 98] },
];

const CLASSES = ['Benign', 'Botnet', 'DDoS', 'DoS', 'PortScan'];

const GaugeChart = ({ value, label, color }: { value: number, label: string, color: string }) => {
  const data = [
    { value: value, color: color },
    { value: 100 - value, color: '#1e293b' },
  ];
  return (
    <div className="flex flex-col items-center">
      <div className="h-32 w-32 relative">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={40}
              outerRadius={50}
              startAngle={180}
              endAngle={0}
              dataKey="value"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        <div className="absolute inset-0 flex flex-col items-center justify-center pt-6">
          <span className="text-xl font-bold text-slate-100">{value}%</span>
        </div>
      </div>
      <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest -mt-4">{label}</span>
    </div>
  );
};

export const ModelRetraining: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-slate-100">Model Health & MLOps Hub</h2>
          <p className="text-slate-400 text-sm">Managing Teacher (CNN-LSTM) and Student (LightGBM) models</p>
        </div>
        <div className="flex gap-4">
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 flex items-center gap-3">
            <div className="p-1.5 bg-blue-500/10 rounded">
              <ShieldCheck size={16} className="text-blue-400" />
            </div>
            <div>
              <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Teacher Model</p>
              <p className="text-xs font-bold text-slate-200">v1.2.0 <span className="text-emerald-400 ml-1">● Active</span></p>
            </div>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 flex items-center gap-3">
            <div className="p-1.5 bg-purple-500/10 rounded">
              <Zap size={16} className="text-purple-400" />
            </div>
            <div>
              <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Student Model</p>
              <p className="text-xs font-bold text-slate-200">v1.2.0 <span className="text-emerald-400 ml-1">● Deployed</span></p>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Gauges */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
          <div className="flex items-center gap-2 mb-6">
            <Target size={18} className="text-blue-400" />
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">Performance Metrics</h3>
          </div>
          <div className="flex justify-around items-center h-[200px]">
            <GaugeChart value={96.5} label="Accuracy" color="#3b82f6" />
            <GaugeChart value={94.2} label="Precision" color="#a855f7" />
            <GaugeChart value={92.8} label="Recall" color="#fbbf24" />
          </div>
        </div>

        {/* Confusion Matrix */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
          <div className="flex items-center gap-2 mb-6">
            <Activity size={18} className="text-blue-400" />
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">Confusion Matrix</h3>
          </div>
          <div className="grid grid-cols-6 gap-1">
            <div className="h-8"></div>
            {CLASSES.map(c => (
              <div key={c} className="h-8 flex items-center justify-center text-[8px] font-bold text-slate-500 uppercase tracking-tighter rotate-45">{c}</div>
            ))}
            {CONFUSION_MATRIX.map((row, i) => (
              <React.Fragment key={i}>
                <div className="h-8 flex items-center justify-end pr-2 text-[8px] font-bold text-slate-500 uppercase tracking-tighter">{row.label}</div>
                {row.values.map((val, j) => (
                  <div 
                    key={j} 
                    className="h-8 rounded-sm flex items-center justify-center text-[10px] font-bold text-slate-100"
                    style={{ backgroundColor: `rgba(59, 130, 246, ${val / 100})` }}
                  >
                    {val}
                  </div>
                ))}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Data Drift */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
          <div className="flex items-center gap-2 mb-6">
            <TrendingUp size={18} className="text-blue-400" />
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">Data Drift Monitor</h3>
          </div>
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={DRIFT_DATA}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis dataKey="time" hide />
                <YAxis stroke="#64748b" fontSize={10} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }} />
                <Line type="monotone" dataKey="drift" stroke="#f43f5e" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="baseline" stroke="#3b82f6" strokeDasharray="5 5" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 flex justify-between text-[10px] font-bold uppercase tracking-widest">
            <span className="text-slate-500">Baseline</span>
            <span className="text-rose-400">Current Drift: 0.24</span>
          </div>
        </div>
      </div>

      {/* Retraining Control */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* Warning Banner */}
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 flex items-center gap-4"
          >
            <div className="p-2 bg-amber-500/20 rounded-full">
              <AlertCircle className="text-amber-500" size={24} />
            </div>
            <div className="flex-1">
              <h4 className="text-amber-500 font-bold text-sm">Model performance degraded (F1-Score -4%)</h4>
              <p className="text-amber-500/70 text-xs">Retraining recommended using 12,450 new verified samples from human-in-the-loop feedback.</p>
            </div>
            <button className="bg-amber-500 hover:bg-amber-400 text-slate-950 px-4 py-2 rounded-lg text-xs font-bold transition-all">
              Auto-Configure
            </button>
          </motion.div>

          {/* Config Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
            <div className="flex items-center gap-2 mb-6">
              <Settings2 size={18} className="text-blue-400" />
              <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">Retraining Configuration</h3>
            </div>
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1.5">Data Batch Range</label>
                  <select className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-300 focus:outline-none focus:border-blue-500">
                    <option>Last 7 Days (12.4K samples)</option>
                    <option>Last 30 Days (45.2K samples)</option>
                    <option>Custom Range...</option>
                  </select>
                </div>
                <div>
                  <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1.5">Training Epochs</label>
                  <input type="number" defaultValue={50} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-300 focus:outline-none focus:border-blue-500" />
                </div>
              </div>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-slate-950/50 border border-slate-800 rounded-lg">
                  <div>
                    <p className="text-xs font-bold text-slate-200">Knowledge Distillation</p>
                    <p className="text-[10px] text-slate-500">Teacher to Student transfer</p>
                  </div>
                  <div className="w-10 h-5 bg-blue-600 rounded-full relative cursor-pointer">
                    <div className="absolute right-1 top-1 w-3 h-3 bg-white rounded-full"></div>
                  </div>
                </div>
                <button className="w-full bg-blue-600 hover:bg-blue-500 text-white py-3 rounded-lg font-bold flex items-center justify-center gap-2 transition-all shadow-lg shadow-blue-600/40 group">
                  <Play size={18} className="group-hover:scale-110 transition-transform" />
                  TRIGGER RETRAINING PIPELINE
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Progress Stepper */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
          <div className="flex items-center gap-2 mb-6">
            <Loader2 size={18} className="text-blue-400 animate-spin" />
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">Pipeline Progress</h3>
          </div>
          <div className="space-y-8 relative before:absolute before:left-4 before:top-2 before:bottom-2 before:w-0.5 before:bg-slate-800">
            <div className="relative pl-10">
              <div className="absolute left-0 top-0 w-8 h-8 bg-emerald-500/20 border border-emerald-500/50 rounded-full flex items-center justify-center z-10">
                <CheckCircle2 size={16} className="text-emerald-400" />
              </div>
              <p className="text-sm font-bold text-slate-200">Extracting Logs</p>
              <p className="text-[10px] text-slate-500 uppercase tracking-widest">Completed</p>
            </div>
            <div className="relative pl-10">
              <div className="absolute left-0 top-0 w-8 h-8 bg-blue-500/20 border border-blue-500 rounded-full flex items-center justify-center z-10">
                <Loader2 size={16} className="text-blue-400 animate-spin" />
              </div>
              <div className="flex justify-between items-end mb-1">
                <p className="text-sm font-bold text-slate-200">Training Teacher</p>
                <span className="text-[10px] font-bold text-blue-400">65%</span>
              </div>
              <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
                <div className="h-full bg-blue-500 w-[65%]"></div>
              </div>
              <p className="text-[10px] text-slate-500 uppercase tracking-widest mt-1">CNN-LSTM Architecture</p>
            </div>
            <div className="relative pl-10">
              <div className="absolute left-0 top-0 w-8 h-8 bg-slate-800 border border-slate-700 rounded-full flex items-center justify-center z-10">
                <div className="w-2 h-2 bg-slate-600 rounded-full"></div>
              </div>
              <p className="text-sm font-bold text-slate-500">Distilling Student</p>
              <p className="text-[10px] text-slate-500 uppercase tracking-widest">Waiting...</p>
            </div>
            <div className="relative pl-10">
              <div className="absolute left-0 top-0 w-8 h-8 bg-slate-800 border border-slate-700 rounded-full flex items-center justify-center z-10">
                <div className="w-2 h-2 bg-slate-600 rounded-full"></div>
              </div>
              <p className="text-sm font-bold text-slate-500">Deploying to Edge</p>
              <p className="text-[10px] text-slate-500 uppercase tracking-widest">Waiting...</p>
            </div>
          </div>
          <div className="mt-8 p-4 bg-slate-950/50 border border-slate-800 rounded-lg">
            <div className="flex items-center gap-2 text-blue-400 mb-1">
              <ArrowRight size={14} />
              <span className="text-[10px] font-bold uppercase tracking-widest">Estimated Time</span>
            </div>
            <p className="text-lg font-bold text-slate-200">12m 45s</p>
          </div>
        </div>
      </div>
    </div>
  );
};
