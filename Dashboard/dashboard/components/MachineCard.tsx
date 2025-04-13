import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { MachineData } from '../lib/useStreamData';
import SensorChart from './SensorChart';
import HealthStatus from './HealthStatus';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

interface MachineCardProps {
  machineName: string;
  machineData: MachineData;
  onMaintenanceNeeded: (machine: string, sensor: string, reason: string, timestamp: string) => void;
}

export default function MachineCard({ machineName, machineData, onMaintenanceNeeded }: MachineCardProps) {
  const [healthHistory, setHealthHistory] = useState<Array<{ timestamp: string; value: number }>>([]);
  const [anomalySensors, setAnomalySensors] = useState<Set<string>>(new Set());
  const sensors = machineData.sensors || {};
  const sensorNames = Object.keys(sensors);
  const prediction = machineData.prediction || null;
  
  // Update health history when new data arrives
  useEffect(() => {
    if (prediction && typeof prediction.predicted_health_index === 'number') {
      setHealthHistory(prev => {
        const newEntry = {
          timestamp: new Date().toLocaleTimeString(),
          value: prediction.predicted_health_index as number
        };
        const updated = [...prev, newEntry].slice(-30); // Keep last 30 points
        return updated;
      });
    }
  }, [prediction]);
  
  // Check if maintenance is needed
  useEffect(() => {
    const newAnomalySensors = new Set<string>();
    
    if (prediction && prediction.risk_score > 0.6) {
      onMaintenanceNeeded(
        machineName,
        'Overall',
        prediction.reason || 'High risk score detected',
        new Date().toISOString()
      );
    }
    
    // Check individual sensors for anomalies (simplified example)
    Object.entries(sensors).forEach(([sensorName, sensorData]) => {
      const lastValue = sensorData.value;
      const history = sensorData.history || [];
      
      if (history.length > 5) {
        const recentValues = history.slice(-5).map(p => p.Value);
        const avg = recentValues.reduce((a, b) => a + b, 0) / recentValues.length;
        const stdDev = Math.sqrt(
          recentValues.map(x => Math.pow(x - avg, 2)).reduce((a, b) => a + b, 0) / recentValues.length
        );
        
        // Simple anomaly detection
        const isAnomaly = Math.abs(lastValue - avg) > stdDev * 2.5;
        if (isAnomaly) {
          newAnomalySensors.add(sensorName);
          
          onMaintenanceNeeded(
            machineName,
            sensorName,
            `Anomaly detected: ${lastValue.toFixed(2)} ${sensorData.unit} (Avg: ${avg.toFixed(2)})`,
            new Date().toISOString()
          );
        }
      }
    });
    
    setAnomalySensors(newAnomalySensors);
  }, [machineData, machineName, onMaintenanceNeeded, sensors]);
  
  // Get status color based on prediction status
  const getStatusColor = () => {
    if (!prediction?.status) return { border: 'border-slate-300', bg: 'bg-slate-100', text: 'text-slate-800', iconBg: 'bg-slate-200', iconText: 'text-slate-500' };
    
    switch (prediction.status.toLowerCase()) {
      case 'healthy':
        return { border: 'border-emerald-300', bg: 'bg-emerald-50', text: 'text-emerald-800', iconBg: 'bg-emerald-100', iconText: 'text-emerald-600' };
      case 'investigate':
        return { border: 'border-blue-300', bg: 'bg-blue-50', text: 'text-blue-800', iconBg: 'bg-blue-100', iconText: 'text-blue-600' };
      case 'warning':
        return { border: 'border-amber-300', bg: 'bg-amber-50', text: 'text-amber-800', iconBg: 'bg-amber-100', iconText: 'text-amber-600' };
      case 'critical':
        return { border: 'border-rose-300', bg: 'bg-rose-50', text: 'text-rose-800', iconBg: 'bg-rose-100', iconText: 'text-rose-600' };
      default:
        return { border: 'border-slate-300', bg: 'bg-slate-100', text: 'text-slate-800', iconBg: 'bg-slate-200', iconText: 'text-slate-500' };
    }
  };
  
  const statusColors = getStatusColor();
  
  return (
    <Link href={`/machines/${encodeURIComponent(machineName)}`} className="block h-full">
      <Card className={`overflow-hidden border ${statusColors.border} bg-white shadow-sm hover:shadow-lg transition-shadow duration-200 h-full flex flex-col`}>
        <CardHeader className={`p-4 border-b ${statusColors.border} bg-gradient-to-r from-white ${statusColors.bg} bg-opacity-30`}>
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <div className={`p-1.5 rounded-full ${statusColors.iconBg}`}>
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={statusColors.iconText}>
                  {prediction?.status?.toLowerCase() === 'healthy' && <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"><polyline points="22 4 12 14.01 9 11.01"></polyline></path>}
                  {prediction?.status?.toLowerCase() === 'investigate' && <><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></>}
                  {prediction?.status?.toLowerCase() === 'warning' && <><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></>}
                  {prediction?.status?.toLowerCase() === 'critical' && <><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></>}
                  {!['healthy', 'investigate', 'warning', 'critical'].includes(prediction?.status?.toLowerCase() || '') && <circle cx="12" cy="12" r="10"></circle>}
                </svg>
              </div>
              <CardTitle className="text-lg font-semibold text-slate-800">{machineName}</CardTitle>
            </div>
            
            <div className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${statusColors.bg} ${statusColors.text}`}>
              {prediction?.status || 'Unknown'}
            </div>
          </div>
        </CardHeader>
        
        <CardContent className="p-4 flex-grow">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 h-full">
            {/* Health Status Panel */}
            <div className="md:col-span-1 flex flex-col">
              <HealthStatus 
                prediction={prediction}
                healthHistory={healthHistory}
              />
            </div>
            
            {/* Sensors Panel */}
            <div className="md:col-span-2 flex flex-col">
              {sensorNames.length > 0 ? (
                <Tabs defaultValue={sensorNames[0]} className="w-full flex flex-col flex-grow">
                  <TabsList className="w-full grid grid-cols-3 gap-1 bg-slate-100 p-1 rounded-lg mb-3 h-auto shrink-0">
                    {sensorNames.map(sensorName => {
                      const isAnomaly = anomalySensors.has(sensorName);
                      return (
                      <TabsTrigger 
                        key={sensorName}
                        value={sensorName}
                        className={`text-xs px-2 py-1.5 rounded-md relative flex flex-col items-center justify-center data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-indigo-700 ${isAnomaly ? 'text-rose-600 font-medium' : 'text-slate-600'}`}
                      >
                        {isAnomaly && (
                          <span className="absolute top-1 right-1 flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500"></span>
                          </span>
                        )}
                        <span className="truncate font-medium">{sensorName}</span>
                        <span className={`text-[11px] ${isAnomaly ? 'text-rose-500' : 'text-slate-500'}`}>
                          {sensors[sensorName].value.toFixed(1)} {sensors[sensorName].unit}
                        </span>
                      </TabsTrigger>
                    )})}
                  </TabsList>
                  
                  <div className="flex-grow pt-1">
                    {sensorNames.map(sensorName => (
                      <TabsContent key={sensorName} value={sensorName} className="p-1 rounded-lg border border-slate-200 bg-white h-full m-0">
                        <SensorChart 
                          sensorName={sensorName}
                          sensorData={sensors[sensorName]}
                          upperBound={
                            sensors[sensorName].history && sensors[sensorName].history.length > 0
                              ? Math.max(...sensors[sensorName].history.map(p => p.Value)) * 1.15
                              : undefined
                          }
                          lowerBound={
                            sensors[sensorName].history && sensors[sensorName].history.length > 0
                              ? Math.min(...sensors[sensorName].history.map(p => p.Value)) * 0.85
                              : undefined
                          }
                          isAnomaly={anomalySensors.has(sensorName)}
                        />
                        
                        <div className="mt-3 p-3 bg-slate-50 rounded-lg border border-slate-100">
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="text-xs font-medium text-slate-600">Current Value</p>
                              <p className={`text-lg font-bold ${anomalySensors.has(sensorName) ? 'text-rose-600' : 'text-slate-900'}`}>
                                {sensors[sensorName].value.toFixed(2)} {sensors[sensorName].unit}
                              </p>
                            </div>
                            
                            {anomalySensors.has(sensorName) && (
                              <div className="bg-rose-100 text-rose-700 px-2 py-1 rounded-md text-xs font-medium flex items-center">
                                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1">
                                  <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                                  <line x1="12" y1="9" x2="12" y2="13"></line>
                                  <line x1="12" y1="17" x2="12.01" y2="17"></line>
                                </svg>
                                Anomaly
                              </div>
                            )}
                          </div>
                        </div>
                      </TabsContent>
                    ))}
                  </div>
                </Tabs>
              ) : (
                <div className="flex items-center justify-center h-full bg-slate-50 rounded-lg border border-dashed border-slate-200 md:col-span-2 min-h-[200px]">
                  <div className="text-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-slate-400 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                    </svg>
                    <p className="text-sm text-slate-500">No sensor data available for this machine.</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
} 