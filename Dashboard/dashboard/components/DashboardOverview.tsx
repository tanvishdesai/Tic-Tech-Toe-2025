import React, { useMemo } from 'react';
import { StreamData } from '../lib/useStreamData';

interface OverviewStats {
  totalMachines: number;
  healthyMachines: number;
  warningMachines: number;
  criticalMachines: number;
  healthyPercentage: number;
}

interface DashboardOverviewProps {
  data: StreamData | null;
}

export default function DashboardOverview({ data }: DashboardOverviewProps) {
  // Calculate overview statistics
  const stats: OverviewStats = useMemo(() => {
    if (!data || !data.machines) {
      return {
        totalMachines: 0,
        healthyMachines: 0,
        warningMachines: 0,
        criticalMachines: 0,
        healthyPercentage: 0
      };
    }

    const machines = Object.entries(data.machines);
    const totalMachines = machines.length;
    
    let healthyMachines = 0;
    let warningMachines = 0;
    let criticalMachines = 0;
    
    machines.forEach(([_, machineData]) => {
      const status = machineData.prediction?.status?.toLowerCase() || '';
      if (status === 'healthy') {
        healthyMachines++;
      } else if (status === 'warning' || status === 'investigate') {
        warningMachines++;
      } else if (status === 'critical') {
        criticalMachines++;
      }
    });
    
    const healthyPercentage = totalMachines > 0 
      ? (healthyMachines / totalMachines) * 100 
      : 0;
      
    return {
      totalMachines,
      healthyMachines,
      warningMachines,
      criticalMachines,
      healthyPercentage
    };
  }, [data]);
  
  // Format the current time
  const currentTime = useMemo(() => {
    return new Date(data?.timestamp || Date.now()).toLocaleString();
  }, [data?.timestamp]);
  
  const stats_cards = [
    {
      title: 'Healthy',
      value: stats.healthyMachines,
      color: 'bg-emerald-500',
      percentValue: stats.healthyPercentage,
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      )
    },
    {
      title: 'Warning',
      value: stats.warningMachines,
      color: 'bg-amber-500',
      percentValue: stats.warningMachines > 0 
        ? (stats.warningMachines / stats.totalMachines) * 100 
        : 0,
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      )
    },
    {
      title: 'Critical',
      value: stats.criticalMachines,
      color: 'bg-rose-500',
      percentValue: stats.criticalMachines > 0 
        ? (stats.criticalMachines / stats.totalMachines) * 100 
        : 0,
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-rose-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      )
    },
    {
      title: 'Total',
      value: stats.totalMachines,
      color: 'bg-blue-500',
      percentValue: 100,
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
        </svg>
      )
    }
  ];
  
  return (
    <div className="bg-white rounded-lg shadow-md p-5 mb-6 border border-gray-100">
      <div className="flex flex-wrap justify-between items-center mb-6">
        <div className="flex items-center">
          <div className="p-2 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-100 mr-3">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-gray-800">Factory Overview</h2>
        </div>
        <div className="flex items-center text-sm text-gray-500 bg-gray-50 px-3 py-1.5 rounded-full border border-gray-200">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>Last updated: {currentTime}</span>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 mb-6">
        {stats_cards.map(card => (
          <div 
            key={card.title} 
            className="bg-white border rounded-lg shadow-sm p-5 flex flex-col transition-all hover:shadow-md"
          >
            <div className="flex items-center mb-3">
              <div className="p-2 rounded-md bg-gray-50 mr-3 border border-gray-100">
                {card.icon}
              </div>
              <span className="text-sm font-medium text-gray-600">{card.title}</span>
            </div>
            <div className="text-3xl font-bold mb-3 text-gray-800">{card.value}</div>
            <div className="mt-auto w-full">
              <div className="flex justify-between text-xs text-gray-500 mb-1">
                <span>0%</span>
                <span>{Math.round(card.percentValue)}%</span>
                <span>100%</span>
              </div>
              <div className="w-full bg-gray-100 rounded-full h-2.5 overflow-hidden shadow-inner">
                <div 
                  className={`h-2.5 rounded-full ${card.color} transition-all duration-500`}
                  style={{ width: `${card.percentValue}%` }}
                >
                  <div className="h-full w-full bg-stripes-white opacity-20"></div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {stats.criticalMachines > 0 ? (
        <div className="p-4 bg-rose-50 rounded-lg border border-rose-200">
          <div className="flex items-center">
            <div className="p-2 bg-rose-100 rounded-full mr-3">
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor" 
                className="h-5 w-5 text-rose-600"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" 
                />
              </svg>
            </div>
            <div>
              <h3 className="font-semibold text-rose-800 mb-0.5">Critical Alert</h3>
              <span className="text-sm text-rose-700">
                {stats.criticalMachines} machine{stats.criticalMachines > 1 ? 's' : ''} in CRITICAL condition. Immediate attention required.
              </span>
            </div>
          </div>
        </div>
      ) : stats.warningMachines > 0 ? (
        <div className="p-4 bg-amber-50 rounded-lg border border-amber-200">
          <div className="flex items-center">
            <div className="p-2 bg-amber-100 rounded-full mr-3">
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor" 
                className="h-5 w-5 text-amber-600"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" 
                />
              </svg>
            </div>
            <div>
              <h3 className="font-semibold text-amber-800 mb-0.5">Warning</h3>
              <span className="text-sm text-amber-700">
                {stats.warningMachines} machine{stats.warningMachines > 1 ? 's' : ''} need investigation.
              </span>
            </div>
          </div>
        </div>
      ) : (
        <div className="p-4 bg-emerald-50 rounded-lg border border-emerald-200">
          <div className="flex items-center">
            <div className="p-2 bg-emerald-100 rounded-full mr-3">
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor" 
                className="h-5 w-5 text-emerald-600"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" 
                />
              </svg>
            </div>
            <div>
              <h3 className="font-semibold text-emerald-800 mb-0.5">All Systems Normal</h3>
              <span className="text-sm text-emerald-700">
                All machines are operating within expected parameters.
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 