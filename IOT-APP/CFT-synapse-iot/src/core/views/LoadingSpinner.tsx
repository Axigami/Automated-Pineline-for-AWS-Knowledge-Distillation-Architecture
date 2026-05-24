import React from 'react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  label?: string;
}

const sizeMap = {
  sm: 'w-4 h-4',
  md: 'w-6 h-6',
  lg: 'w-10 h-10',
};

/**
 * LoadingSpinner – Atom dùng chung cho trạng thái loading.
 */
export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  label,
}) => {
  return (
    <div className="flex flex-col items-center justify-center gap-2">
      <div
        className={`${sizeMap[size]} border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin`}
      />
      {label && <span className="text-xs text-slate-400">{label}</span>}
    </div>
  );
};
