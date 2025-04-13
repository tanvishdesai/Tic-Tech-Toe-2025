import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, ReferenceArea } from 'recharts';
import { PredictionData } from '../lib/useStreamData';

interface HealthStatusProps {
  prediction?: PredictionData | null;
  healthHistory?: Array<{ timestamp: string; value: number }>;
}

export default function HealthStatus({ prediction, healthHistory = [] }: HealthStatusProps) {
  // Default values if no prediction is available
  const status = prediction?.status || 'Unknown';
  const healthIndex = prediction?.predicted_health_index || 0;
  const riskScore = prediction?.risk_score || 0;
  const reason = prediction?.reason || 'No data available';

  // Determine status color
  let statusColor = 'bg-gray-500';
  let textColor = 'text-gray-700';
  let borderColor = 'border-gray-300';
  let iconType = 'question-mark';
  
  switch (status.toLowerCase()) {
    case 'healthy':
      statusColor = 'bg-emerald-500';
      textColor = 'text-emerald-700';
      borderColor = 'border-emerald-200';
      iconType = 'check-circle';
      break;
    case 'investigate':
      statusColor = 'bg-blue-500';
      textColor = 'text-blue-700';
      borderColor = 'border-blue-200';
      iconType = 'search';
      break;
    case 'warning':
    case 'anomaly detected':
      statusColor = 'bg-amber-500';
      textColor = 'text-amber-700';
      borderColor = 'border-amber-200';
      iconType = 'alert-triangle';
      break;
    case 'critical':
      statusColor = 'bg-rose-500';
      textColor = 'text-rose-700';
      borderColor = 'border-rose-200';
      iconType = 'alert-octagon';
      break;
  }

  // Calculate health indicator percentage for visual display
  const healthPercent = healthIndex !== null ? Math.max(0, Math.min(100, healthIndex * 100)) : 50;
  
  return (
    <div className={`bg-white p-5 rounded-lg shadow border ${borderColor}`}>
      <div className="flex items-center mb-5">
        <div className={`p-2 rounded-full ${statusColor} bg-opacity-20 mr-3`}>
          {iconType === 'check-circle' && (
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={textColor}>
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
              <polyline points="22 4 12 14.01 9 11.01"></polyline>
            </svg>
          )}
          {iconType === 'search' && (
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={textColor}>
              <circle cx="11" cy="11" r="8"></circle>
              <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
          )}
          {iconType === 'alert-triangle' && (
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={textColor}>
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
              <line x1="12" y1="9" x2="12" y2="13"></line>
              <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
          )}
          {iconType === 'alert-octagon' && (
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={textColor}>
              <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon>
              <line x1="12" y1="8" x2="12" y2="12"></line>
              <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
          )}
          {iconType === 'question-mark' && (
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={textColor}>
              <circle cx="12" cy="12" r="10"></circle>
              <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
              <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
          )}
        </div>
        <div>
          <h3 className="font-medium text-lg">Health Status</h3>
          <div className={`text-sm font-medium ${textColor}`}>{status}</div>
        </div>
      </div>
      
      {/* Health Index Visual */}
      <div className="mb-5">
        <div className="flex justify-between items-center mb-1.5">
          <span className="text-sm font-medium text-gray-700">Health Index</span>
          <span className="text-sm font-bold">{(healthIndex * 100).toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-100 rounded-full h-3 overflow-hidden shadow-inner">
          <div 
            className={`h-3 rounded-full transition-all duration-500 ${
              healthPercent > 80 ? 'bg-emerald-500' : 
              healthPercent > 50 ? 'bg-blue-500' : 
              healthPercent > 30 ? 'bg-amber-500' : 
              'bg-rose-500'
            }`}
            style={{ width: `${healthPercent}%` }}
          >
            <div className="h-full w-full bg-stripes-white opacity-20"></div>
          </div>
        </div>
      </div>
      
      {/* Risk Score Visual */}
      <div className="mb-5">
        <div className="flex justify-between items-center mb-1.5">
          <span className="text-sm font-medium text-gray-700">Risk Score</span>
          <span className="text-sm font-bold">{riskScore.toFixed(2)}</span>
        </div>
        <div className="w-full bg-gray-100 rounded-full h-3 overflow-hidden shadow-inner">
          <div 
            className={`h-3 rounded-full transition-all duration-500 ${
              riskScore > 0.7 ? 'bg-rose-500' : 
              riskScore > 0.4 ? 'bg-amber-500' : 
              riskScore > 0.2 ? 'bg-blue-500' : 
              'bg-emerald-500'
            }`}
            style={{ width: `${Math.min(100, riskScore * 100)}%` }}
          >
            <div className="h-full w-full bg-stripes-white opacity-20"></div>
          </div>
        </div>
      </div>
      
      {/* Reason for status */}
      <div className={`text-sm p-3 rounded-lg ${
        status.toLowerCase() === 'critical' ? 'bg-rose-50 text-rose-800 border border-rose-100' :
        status.toLowerCase() === 'warning' ? 'bg-amber-50 text-amber-800 border border-amber-100' :
        status.toLowerCase() === 'investigate' ? 'bg-blue-50 text-blue-800 border border-blue-100' :
        'bg-gray-50 text-gray-800 border border-gray-100'
      }`}>
        <p>{reason}</p>
      </div>
      
      {/* Health History Chart */}
      {healthHistory && healthHistory.length > 0 && (
        <div className="h-40 mt-5">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={healthHistory} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis 
                dataKey="timestamp" 
                tick={{ fontSize: 10 }}
                minTickGap={15}
                stroke="#9ca3af"
              />
              <YAxis 
                domain={[0, 1]} 
                tick={{ fontSize: 10 }} 
                tickFormatter={(value) => (value * 100).toFixed(0) + '%'}
                stroke="#9ca3af"
              />
              <Tooltip 
                formatter={(value: number) => [(value * 100).toFixed(1) + '%', 'Health Index']}
                labelFormatter={(label) => `Time: ${label}`}
                contentStyle={{ borderRadius: '8px', border: '1px solid #e5e7eb' }}
              />
              
              {/* Critical threshold area */}
              <ReferenceArea y1={0} y2={0.2} fill="#fecaca" fillOpacity={0.3} />
              
              {/* Warning threshold area */}
              <ReferenceArea y1={0.2} y2={0.5} fill="#fed7aa" fillOpacity={0.2} />
              
              {/* Healthy threshold line */}
              <ReferenceLine 
                y={0.8} 
                stroke="#10b981" 
                strokeDasharray="3 3" 
                label={{ value: 'Optimal', position: 'right', fontSize: 10, fill: '#10b981' }}
              />
              
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#3b82f6" 
                strokeWidth={2}
                dot={{ r: 0 }}
                activeDot={{ r: 4, stroke: '#3b82f6', strokeWidth: 1, fill: '#fff' }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
} 