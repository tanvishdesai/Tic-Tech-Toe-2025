import React, { useMemo } from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  ReferenceLine,
  Area,
  ComposedChart,
  Scatter,
  ReferenceArea,
  Legend
} from 'recharts';
import { SensorData } from '../lib/useStreamData';

interface SensorChartProps {
  sensorName: string;
  sensorData: SensorData;
  upperBound?: number;
  lowerBound?: number;
  isAnomaly?: boolean; // Overall anomaly status for the sensor
}

// Custom Tooltip Component
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const dataPoint = payload[0].payload;
    const value = dataPoint.value;
    const unit = payload[0].payload.unit || ''; // Assuming unit might be available
    const isPointAnomaly = dataPoint.anomaly !== null;

    return (
      <div className="bg-white p-2 rounded-lg shadow-lg border border-slate-200 text-xs">
        <p className="text-slate-500 mb-1">{`Time: ${label}`}</p>
        <p className={`font-medium ${isPointAnomaly ? 'text-rose-600' : 'text-slate-800'}`}>
          {`${payload[0].name}: ${value.toFixed(2)} ${unit}`}
        </p>
        {payload.find((p: any) => p.dataKey === 'mean') && (
           <p className="text-slate-600">
            {`Rolling Avg: ${dataPoint.mean.toFixed(2)} ${unit}`}
          </p>
        )}
        {isPointAnomaly && (
          <p className="text-rose-600 font-semibold mt-1">Anomaly Detected</p>
        )}
      </div>
    );
  }
  return null;
};

export default function SensorChart({ sensorName, sensorData, upperBound, lowerBound, isAnomaly }: SensorChartProps) {
  if (!sensorData || !sensorData.history || sensorData.history.length === 0) {
    return <div className="flex items-center justify-center h-64 bg-slate-50 rounded-lg text-slate-500 text-sm">No historical data available</div>;
  }

  const data = useMemo(() => {
    return sensorData.history.map((point, index, array) => {
      // Simplified anomaly detection logic (assuming server-side detection is primary)
      // You might want to refine this or rely solely on flags from the server
      const windowSize = 5;
      const startIdx = Math.max(0, index - windowSize);
      const endIdx = Math.min(array.length - 1, index + windowSize);
      const window = array.slice(startIdx, endIdx + 1).map(p => p.Value);
      const sum = window.reduce((acc, curr) => acc + curr, 0);
      const mean = sum / window.length;
      const stdDev = Math.sqrt(
        window.reduce((acc, curr) => acc + Math.pow(curr - mean, 2), 0) / window.length
      );
      const isPointAnomaly = Math.abs(point.Value - mean) > stdDev * 2.5; // Example threshold
      
      return {
        timestamp: new Date(point.Timestamp).toLocaleTimeString([], { hour: '2-digit', minute:'2-digit', second: '2-digit' }), // Cleaner time format
        value: point.Value,
        mean,
        unit: sensorData.unit, // Pass unit along
        anomaly: isPointAnomaly ? point.Value : null, // Mark individual anomaly points
      };
    });
  }, [sensorData.history, sensorData.unit]);

  const anomalies = useMemo(() => {
    return data.filter(point => point.anomaly !== null);
  }, [data]);

  const values = data.map(d => d.value);
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  
  const padding = (maxVal - minVal) * 0.20; // Increased padding
  const yMin = Math.floor(Math.max(0, minVal - padding)); // Floor for cleaner axis
  const yMax = Math.ceil(maxVal + padding); // Ceil for cleaner axis

  return (
    // Add subtle red background tint if the sensor is in an overall anomalous state
    <div className={`h-64 w-full rounded-lg ${isAnomaly ? 'bg-rose-50/50' : 'bg-white'} transition-colors duration-300`}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 5, right: 5, bottom: 0, left: -25 }}> {/* Adjusted margins */}
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" opacity={0.6} />
          <XAxis 
            dataKey="timestamp" 
            tick={{ fontSize: 10, fill: '#64748b' }} 
            tickCount={6}
            minTickGap={20}
            stroke="#cbd5e1"
            dy={5} // Adjust tick position
          />
          <YAxis 
            domain={[yMin, yMax]}
            tick={{ fontSize: 10, fill: '#64748b' }}
            tickFormatter={(value) => value.toFixed(0)} // Simpler tick format
            width={30} // Give Y-axis some space
            stroke="#cbd5e1"
            // Label removed, unit added to tooltip
          />
          <Tooltip 
            content={<CustomTooltip />}
            cursor={{ stroke: '#4f46e5', strokeWidth: 1, strokeDasharray: '3 3' }}
          />
          {/* Legend removed for cleaner look, info is in tooltip */}
          
          {/* Optional: Less prominent bounds */}
          {upperBound && (
            <ReferenceLine 
              y={upperBound} 
              stroke="#f59e0b" // Amber color
              strokeOpacity={0.6}
              strokeDasharray="4 4" 
              // label={{ value: 'Upper', position: 'insideTopRight', fontSize: 9, fill: '#a16207', dy: -5 }}
            />
          )}
          {lowerBound && (
            <ReferenceLine 
              y={lowerBound} 
              stroke="#f59e0b" // Amber color
              strokeOpacity={0.6}
              strokeDasharray="4 4" 
              // label={{ value: 'Lower', position: 'insideTopRight', fontSize: 9, fill: '#a16207', dy: 15 }}
            />
          )}
          
          {/* Rolling Average Area (more subtle) */}
          <Area 
            type="monotone" 
            dataKey="mean" 
            name="Rolling Average"
            fill="#e0e7ff" // Lighter Indigo fill
            stroke="#a5b4fc" // Lighter Indigo stroke
            strokeWidth={1}
            dot={false}
            activeDot={false}
            legendType="none" // Hide from legend if shown
          />
          
          {/* Main Data Line (thicker, Indigo) */}
          <Line 
            type="monotone" 
            dataKey="value" 
            name={sensorName}
            stroke="#4f46e5" // Indigo color
            strokeWidth={2.5} // Slightly thicker line
            dot={false} // No dots on the main line
            activeDot={{ r: 4, stroke: '#4f46e5', strokeWidth: 1, fill: '#fff' }}
            isAnimationActive={false} // Disable animation for performance
            legendType="none" // Hide from legend if shown
          />
          
          {/* Anomaly Scatter Points (clearer) */}
          <Scatter 
            dataKey="anomaly" 
            name="Anomaly"
            fill="#dc2626" // Stronger Red
            shape="circle" 
            legendType="none" // Hide from legend if shown
          />
          
          {/* Removed ReferenceArea for anomalies to reduce visual clutter, relying on Scatter points */}

        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
} 