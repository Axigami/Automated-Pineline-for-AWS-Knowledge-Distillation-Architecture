import React, { useState } from 'react';
import { X, Server, MapPin, Network, Cpu } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { toast } from './Toast';

interface AddNodeModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (payload: {
    nodeCode: string;
    location: string;
    ipAddress: string;
    framework: string;
  }) => Promise<void>;
}

export const AddNodeModal: React.FC<AddNodeModalProps> = ({ isOpen, onClose, onSubmit }) => {
  const [nodeCode, setNodeCode] = useState('');
  const [location, setLocation] = useState('');
  const [ipAddress, setIpAddress] = useState('');
  const [framework, setFramework] = useState('ONNX Runtime');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Clear form on close
  const handleClose = () => {
    setNodeCode('');
    setLocation('');
    setIpAddress('');
    setFramework('ONNX Runtime');
    onClose();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!nodeCode || !location) {
      toast('warning', 'Missing Fields', 'Please provide both Node Code and Location.');
      return;
    }
    
    setIsSubmitting(true);
    try {
      await onSubmit({ nodeCode, location, ipAddress, framework });
      handleClose();
      toast('success', 'Node Deployed', `Successfully provisioned edge node ${nodeCode}!`);
    } catch (err: any) {
      toast('error', 'Provisioning Failed', err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop kính mờ */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-[9998]"
            onClick={handleClose}
          />
          
          {/* Modal Container */}
          <div className="fixed inset-0 flex items-center justify-center p-4 z-[9999] pointer-events-none">
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 10 }}
              className="bg-slate-900 border border-slate-700 w-full max-w-md rounded-2xl shadow-2xl overflow-hidden pointer-events-auto flex flex-col"
            >
              {/* Header */}
              <div className="px-6 py-4 border-b border-slate-800 flex justify-between items-center bg-slate-800/50">
                <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                  <Server size={18} className="text-blue-500" />
                  Add New Edge Node
                </h3>
                <button 
                  onClick={handleClose}
                  className="text-slate-400 hover:text-slate-200 transition-colors p-1"
                >
                  <X size={20} />
                </button>
              </div>

              {/* Body Form */}
              <form onSubmit={handleSubmit} className="p-6 space-y-4">
                {/* Node Code */}
                <div className="space-y-1">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                    <Server size={12} /> Node Code <span className="text-red-500">*</span>
                  </label>
                  <input 
                    type="text" 
                    placeholder="E.g., EDGE005"
                    value={nodeCode}
                    onChange={e => setNodeCode(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-700 text-slate-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all font-mono placeholder:text-slate-600"
                    required
                  />
                </div>

                {/* Location */}
                <div className="space-y-1">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                    <MapPin size={12} /> Location <span className="text-red-500">*</span>
                  </label>
                  <input 
                    type="text" 
                    placeholder="E.g., Cần Thơ"
                    value={location}
                    onChange={e => setLocation(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-700 text-slate-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all placeholder:text-slate-600"
                    required
                  />
                  <p className="text-[10px] text-slate-500">The fleet map will auto-locate this position.</p>
                </div>

                {/* IP Address */}
                <div className="space-y-1">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                    <Network size={12} /> IP / Tên miền mạng
                  </label>
                  <input 
                    type="text" 
                    placeholder="192.168.1.x"
                    value={ipAddress}
                    onChange={e => setIpAddress(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-700 text-slate-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all font-mono placeholder:text-slate-600"
                  />
                </div>

                {/* Framework */}
                <div className="space-y-1">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                    <Cpu size={12} /> AI Framework
                  </label>
                  <select
                    value={framework}
                    onChange={e => setFramework(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-700 text-slate-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all"
                  >
                    <option value="ONNX Runtime">ONNX Runtime</option>
                    <option value="TensorFlow Lite">TensorFlow Lite</option>
                    <option value="PyTorch Mobile">PyTorch Mobile</option>
                    <option value="NVIDIA TensorRT">NVIDIA TensorRT</option>
                  </select>
                </div>

                {/* Footer buttons */}
                <div className="pt-4 mt-2 border-t border-slate-800 flex justify-end gap-3">
                  <button
                    type="button"
                    onClick={handleClose}
                    className="px-4 py-2 rounded-lg text-sm font-bold text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={isSubmitting}
                    className="px-4 py-2 rounded-lg text-sm font-bold text-white bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-500/20"
                  >
                    {isSubmitting ? 'Provisioning...' : 'Deploy Node'}
                  </button>
                </div>
              </form>
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
};
