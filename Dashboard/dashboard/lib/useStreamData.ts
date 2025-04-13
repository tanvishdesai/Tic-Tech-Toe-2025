import { useState, useEffect } from 'react';

export interface SensorData {
  value: number;
  unit: string;
  history: {
    Timestamp: string;
    Value: number;
  }[];
}

export interface PredictionData {
  status: string;
  reason: string;
  risk_score: number;
  predicted_health_index: number | null;
}

export interface MachineData {
  sensors: Record<string, SensorData>;
  prediction: PredictionData | null;
}

export interface StreamData {
  timestamp: string;
  machines: Record<string, MachineData>;
}

export function useStreamData() {
  const [data, setData] = useState<StreamData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    let eventSource: EventSource;
    
    const connectToStream = () => {
      setError(null);
      
      eventSource = new EventSource('/api/stream');
      
      eventSource.onopen = () => {
        setIsConnected(true);
        console.log('Connected to event stream');
      };
      
      eventSource.onmessage = (event) => {
        try {
          const parsedData = JSON.parse(event.data);
          setData(parsedData);
        } catch (err) {
          console.error('Error parsing stream data:', err);
          setError('Failed to process data from server');
        }
      };
      
      eventSource.onerror = (err) => {
        console.error('EventSource error:', err);
        setIsConnected(false);
        setError('Connection to data stream failed');
        eventSource.close();
        
        // Attempt to reconnect after a delay
        setTimeout(connectToStream, 5000);
      };
      
      // Listen for specific error events from the backend
      eventSource.addEventListener('error', (event) => {
        const data = JSON.parse((event as any).data);
        setError(`Stream error: ${data.message}`);
      });
      
      eventSource.addEventListener('backend_error', (event) => {
        const data = JSON.parse((event as any).data);
        console.error('Backend error:', data.message);
      });
      
      eventSource.addEventListener('close', (event) => {
        const data = JSON.parse((event as any).data);
        console.log('Stream closed:', data.message);
        setIsConnected(false);
      });
    };
    
    connectToStream();
    
    return () => {
      if (eventSource) {
        eventSource.close();
        setIsConnected(false);
      }
    };
  }, []);
  
  return { data, error, isConnected };
} 