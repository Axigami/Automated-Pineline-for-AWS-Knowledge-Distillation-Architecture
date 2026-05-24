import React from 'react';

interface BaseLayoutProps {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
}

/**
 * BaseLayout – Atom/Molecule cấp toàn cục.
 * Mọi trang module đều bọc nội dung trong component này.
 */
export const BaseLayout: React.FC<BaseLayoutProps> = ({
  title,
  subtitle,
  actions,
  children,
}) => {
  return (
    <div className="flex flex-col gap-6 h-full">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-100">{title}</h2>
          {subtitle && (
            <p className="text-sm text-slate-400 mt-0.5">{subtitle}</p>
          )}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>

      {/* Content */}
      <div className="flex-1">{children}</div>
    </div>
  );
};
