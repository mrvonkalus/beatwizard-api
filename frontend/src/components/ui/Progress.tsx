import React from 'react';
import { cn } from '../../utils';

interface ProgressProps {
  value: number; // 0-100
  size?: 'sm' | 'md' | 'lg';
  color?: 'blue' | 'green' | 'purple' | 'pink' | 'yellow' | 'red';
  showLabel?: boolean;
  label?: string;
  animated?: boolean;
  className?: string;
}

export function Progress({
  value,
  size = 'md',
  color = 'purple',
  showLabel = false,
  label,
  animated = false,
  className
}: ProgressProps) {
  const clampedValue = Math.min(100, Math.max(0, value));
  
  const sizeClasses = {
    sm: 'h-2',
    md: 'h-3',
    lg: 'h-4'
  };
  
  const colorClasses = {
    blue: 'bg-blue-600',
    green: 'bg-green-600',
    purple: 'bg-purple-600',
    pink: 'bg-pink-600',
    yellow: 'bg-yellow-600',
    red: 'bg-red-600'
  };

  return (
    <div className={cn('w-full', className)}>
      {(showLabel || label) && (
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700">
            {label || 'Progress'}
          </span>
          <span className="text-sm text-gray-500">
            {Math.round(clampedValue)}%
          </span>
        </div>
      )}
      
      <div className={cn(
        'bg-gray-200 rounded-full overflow-hidden',
        sizeClasses[size]
      )}>
        <div
          className={cn(
            'h-full rounded-full transition-all duration-300 ease-out',
            colorClasses[color],
            animated && 'animate-pulse'
          )}
          style={{ width: `${clampedValue}%` }}
        />
      </div>
    </div>
  );
}

interface CircularProgressProps {
  value: number; // 0-100
  size?: number;
  strokeWidth?: number;
  color?: string;
  trackColor?: string;
  showLabel?: boolean;
  label?: string;
  className?: string;
}

export function CircularProgress({
  value,
  size = 80,
  strokeWidth = 8,
  color = '#8b5cf6',
  trackColor = '#e5e7eb',
  showLabel = true,
  label,
  className
}: CircularProgressProps) {
  const clampedValue = Math.min(100, Math.max(0, value));
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (clampedValue / 100) * circumference;

  return (
    <div className={cn('relative inline-flex items-center justify-center', className)}>
      <svg
        width={size}
        height={size}
        className="transform -rotate-90"
      >
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={trackColor}
          strokeWidth={strokeWidth}
          fill="transparent"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={color}
          strokeWidth={strokeWidth}
          fill="transparent"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-300 ease-out"
        />
      </svg>
      
      {showLabel && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-medium text-gray-700">
            {label || `${Math.round(clampedValue)}%`}
          </span>
        </div>
      )}
    </div>
  );
}

interface StepProgressProps {
  currentStep: number;
  totalSteps: number;
  steps: Array<{
    label: string;
    description?: string;
    icon?: React.ReactNode;
  }>;
  variant?: 'horizontal' | 'vertical';
  className?: string;
}

export function StepProgress({
  currentStep,
  totalSteps,
  steps,
  variant = 'horizontal',
  className
}: StepProgressProps) {
  const isHorizontal = variant === 'horizontal';
  
  return (
    <div className={cn(
      'flex',
      isHorizontal ? 'items-center space-x-4' : 'flex-col space-y-4',
      className
    )}>
      {steps.slice(0, totalSteps).map((step, index) => {
        const stepNumber = index + 1;
        const isCompleted = stepNumber < currentStep;
        const isCurrent = stepNumber === currentStep;
        const isUpcoming = stepNumber > currentStep;
        
        return (
          <div
            key={stepNumber}
            className={cn(
              'flex items-center',
              !isHorizontal && 'w-full'
            )}
          >
            {/* Step indicator */}
            <div className="relative">
              <div className={cn(
                'flex items-center justify-center w-8 h-8 rounded-full border-2 transition-colors',
                isCompleted && 'bg-green-600 border-green-600',
                isCurrent && 'bg-purple-600 border-purple-600',
                isUpcoming && 'bg-gray-100 border-gray-300'
              )}>
                {isCompleted ? (
                  <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                ) : (
                  <span className={cn(
                    'text-sm font-medium',
                    isCurrent && 'text-white',
                    isUpcoming && 'text-gray-500'
                  )}>
                    {stepNumber}
                  </span>
                )}
              </div>
            </div>
            
            {/* Step content */}
            <div className={cn(
              'ml-3',
              !isHorizontal && 'flex-1'
            )}>
              <p className={cn(
                'text-sm font-medium',
                isCompleted && 'text-green-600',
                isCurrent && 'text-purple-600',
                isUpcoming && 'text-gray-500'
              )}>
                {step.label}
              </p>
              {step.description && (
                <p className="text-xs text-gray-500 mt-1">
                  {step.description}
                </p>
              )}
            </div>
            
            {/* Connector line (horizontal only) */}
            {isHorizontal && index < totalSteps - 1 && (
              <div className={cn(
                'flex-1 h-0.5 mx-4',
                stepNumber < currentStep ? 'bg-green-600' : 'bg-gray-300'
              )} />
            )}
          </div>
        );
      })}
    </div>
  );
}