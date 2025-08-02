import React from 'react';
import { cn } from '../../utils';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  children: React.ReactNode;
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info' | 'purple';
  size?: 'sm' | 'md' | 'lg';
  rounded?: boolean;
}

export function Badge({
  children,
  variant = 'default',
  size = 'md',
  rounded = true,
  className,
  ...props
}: BadgeProps) {
  const baseClasses = 'inline-flex items-center font-medium border';
  
  const variants = {
    default: 'bg-gray-100 text-gray-800 border-gray-200',
    success: 'bg-green-100 text-green-800 border-green-200',
    warning: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    error: 'bg-red-100 text-red-800 border-red-200',
    info: 'bg-blue-100 text-blue-800 border-blue-200',
    purple: 'bg-purple-100 text-purple-800 border-purple-200'
  };
  
  const sizes = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-2.5 py-1.5 text-sm',
    lg: 'px-3 py-2 text-base'
  };

  return (
    <span
      className={cn(
        baseClasses,
        variants[variant],
        sizes[size],
        rounded && 'rounded-full',
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
}

interface QualityBadgeProps {
  quality: string;
  size?: 'sm' | 'md' | 'lg';
  showEmoji?: boolean;
}

export function QualityBadge({ quality, size = 'md', showEmoji = true }: QualityBadgeProps) {
  const getVariant = (q: string) => {
    switch (q.toLowerCase()) {
      case 'excellent':
      case 'good':
        return 'success';
      case 'fair':
      case 'average':
        return 'warning';
      case 'poor':
      case 'needs_improvement':
        return 'error';
      default:
        return 'default';
    }
  };

  const getEmoji = (q: string) => {
    switch (q.toLowerCase()) {
      case 'excellent':
        return '🔥';
      case 'good':
        return '✅';
      case 'fair':
      case 'average':
        return '⚡';
      case 'poor':
      case 'needs_improvement':
        return '⚠️';
      default:
        return '📊';
    }
  };

  return (
    <Badge variant={getVariant(quality)} size={size}>
      {showEmoji && (
        <span className="mr-1">{getEmoji(quality)}</span>
      )}
      {quality.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
    </Badge>
  );
}