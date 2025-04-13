'use client';

import React, { useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import DashboardOverview from '@/components/DashboardOverview';
import MaintenanceLog from '@/components/MaintenanceLog';
import { useStreamData } from '@/lib/useStreamData';

// Dynamically import MachineCard to avoid SSR issues with charts
const MachineCard = dynamic(() => import('@/components/MachineCard'), { 
  ssr: false,
  loading: () => <div className="bg-white rounded-lg shadow p-6 animate-pulse h-64"></div>
});

interface MaintenanceItem {
  id: string;
  machine: string;
  sensor: string;
  timestamp: string;
  reason: string;
  status: 'Pending' | 'Acknowledged' | 'Scheduled' | 'Completed';
}

export default function Home() {
  const { data, error, isConnected } = useStreamData();
  const [maintenanceLogs, setMaintenanceLogs] = useState<MaintenanceItem[]>([]);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'maintenance'>('dashboard');

  // Handle maintenance log status changes
  const handleMaintenanceStatusChange = useCallback((id: string, newStatus: MaintenanceItem['status'], additionalData?: Partial<MaintenanceItem>) => {
    setMaintenanceLogs(prev => 
      prev.map(log => log.id === id ? { ...log, status: newStatus, ...additionalData } : log)
    );
  }, []);

  // Handle new maintenance items
  const handleMaintenanceNeeded = useCallback((machine: string, sensor: string, reason: string, timestamp: string) => {
    // Check if a similar maintenance item already exists to avoid duplicates
    const existingLog = maintenanceLogs.find(log => 
      log.machine === machine && 
      log.sensor === sensor && 
      log.status === 'Pending'
    );
    
    if (!existingLog) {
      const newLog: MaintenanceItem = {
        id: `${machine}-${sensor}-${Date.now()}`,
        machine,
        sensor,
        timestamp,
        reason,
        status: 'Pending'
      };
      
      setMaintenanceLogs(prev => [newLog, ...prev]);
    }
  }, [maintenanceLogs]);

  // Count pending maintenance items
  const pendingMaintenanceCount = maintenanceLogs.filter(log => log.status === 'Pending').length;

  return (
    <div className="min-h-screen bg-slate-100 pb-8">
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10 shadow-sm">
        <div className="container mx-auto px-4 py-3">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7 text-indigo-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
              </svg>
              <h1 className="text-xl font-semibold text-slate-800">Predictive Maintenance</h1>
            </div>

            <div className="flex items-center space-x-4">
              <div className={`flex items-center px-3 py-1 rounded-full text-xs font-medium border ${ 
                isConnected 
                ? 'bg-emerald-50 text-emerald-700 border-emerald-200' 
                : 'bg-amber-50 text-amber-700 border-amber-200 animate-pulse'
              }`}>
                <span className={`h-2 w-2 mr-1.5 rounded-full ${isConnected ? 'bg-emerald-500' : 'bg-amber-500'}`}></span>
                {isConnected ? 'Connected' : 'Connecting...'}
              </div>
              
              <button 
                className="bg-indigo-600 hover:bg-indigo-700 text-white rounded-md px-3 py-1.5 text-sm font-medium flex items-center transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
                onClick={() => window.location.reload()}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Refresh
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 pt-8">
        {error && (
          <div className="bg-rose-50 border border-rose-200 text-rose-700 px-4 py-3 rounded-md relative mb-6 shadow-sm" role="alert">
            <strong className="font-semibold">Connection Error:</strong>
            <span className="block sm:inline ml-2">{error}</span>
          </div>
        )}
        
        {/* Dashboard Overview */}
        <DashboardOverview data={data} />
        
        {/* Tab Navigation - Updated Style */}
        <div className="mb-6 border-b border-slate-200">
          <nav className="-mb-px flex space-x-6" aria-label="Tabs">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`${ 
                activeTab === 'dashboard'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
              } whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm flex items-center`}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                 <path strokeLinecap="round" strokeLinejoin="round" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
              </svg>
              Dashboard
            </button>
            <button
              onClick={() => setActiveTab('maintenance')}
              className={`${ 
                activeTab === 'maintenance'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
              } whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm flex items-center`}
            >
               <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
              Maintenance Log
              {pendingMaintenanceCount > 0 && (
                <span className="ml-2 bg-rose-100 text-rose-600 text-xs font-semibold px-2 py-0.5 rounded-full">
                  {pendingMaintenanceCount}
                </span>
              )}
            </button>
          </nav>
        </div>
        
        {/* Tab Content Area - Added padding top */}
        <div className="pt-4">
          {activeTab === 'dashboard' ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-2 gap-8">
              {data && data.machines && Object.keys(data.machines).length > 0 ? (
                Object.entries(data.machines).map(([machineName, machineData]) => (
                  <MachineCard 
                    key={machineName} 
                    machineName={machineName} 
                    machineData={machineData}
                    onMaintenanceNeeded={handleMaintenanceNeeded}
                  />
                ))
              ) : (
                <div className="lg:col-span-2 xl:col-span-3 text-center py-16 bg-white rounded-lg border border-slate-200 shadow-sm">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-slate-300 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                  <h3 className="text-lg font-medium text-slate-700 mb-1">Waiting for machine data...</h3>
                  <p className="text-slate-500 text-sm max-w-md mx-auto">
                    Connect to the data stream to view real-time machine status and sensor readings.
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-6">
              <MaintenanceLog 
                logs={maintenanceLogs} 
                onStatusChange={handleMaintenanceStatusChange}
              />
            </div>
          )}
        </div>
      </main>
    </div>
  );
} 