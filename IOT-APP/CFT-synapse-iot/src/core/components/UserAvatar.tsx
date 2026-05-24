import React from 'react';

type Size = 'sm' | 'md' | 'lg';

const sizeClasses: Record<Size, { wrap: string; text: string }> = {
  sm: { wrap: 'w-8 h-8 min-w-8 min-h-8', text: 'text-sm' },
  md: { wrap: 'w-10 h-10 min-w-10 min-h-10', text: 'text-base' },
  lg: { wrap: 'w-12 h-12 min-w-12 min-h-12', text: 'text-lg' },
};

interface UserAvatarProps {
  email?: string | null;
  size?: Size;
  className?: string;
}

/**
 * Avatar đồng bộ sidebar / header: chữ cái đầu email, viền xanh giống UserDropdown.
 */
export const UserAvatar: React.FC<UserAvatarProps> = ({
  email,
  size = 'md',
  className = '',
}) => {
  const letter = (email?.trim().charAt(0) || '?').toUpperCase();
  const { wrap, text } = sizeClasses[size];

  return (
    <div
      className={`${wrap} rounded-full bg-blue-600/20 border border-blue-500/30 flex items-center justify-center overflow-hidden flex-shrink-0 ${className}`}
      aria-hidden
    >
      <span className={`text-blue-400 font-bold uppercase select-none ${text}`}>{letter}</span>
    </div>
  );
};
