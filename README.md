# Predictive Maintenance for Smart Manufacturing

A solution designed to analyze equipment sensor data in real time to predict potential failures before they occur. It uses anomaly detection and predictive forecasting to minimize downtime and improve factory efficiency.

## Key Features

- **Real-time Sensor Data Analysis**: Monitor multiple machines and their sensors in real-time
- **Anomaly Detection**: Automatically identify abnormal sensor readings
- **Predictive Maintenance**: Forecast potential failures before they occur
- **Maintenance Scheduling**: Plan and track maintenance activities
- **Interactive Dashboard**: Visualize trends and sensor data

## Tech Stack

### Frontend
- Next.js
- TypeScript
- Tailwind CSS
- shadcn/ui
- Recharts for data visualization
- WebSocket for real-time data updates

### Backend
- FastAPI
- Python
- WebSockets for real-time communication
- Pandas/NumPy for data analysis

## Getting Started

### Running the Backend

1. Navigate to the API directory:
```
cd api
```

2. Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

4. Start the FastAPI server:
```
python main.py
```

The API will be available at http://localhost:8000, with WebSocket connections at ws://localhost:8000/ws.

### Running the Frontend

1. In the project root directory:
```
npm install
```

2. Start the development server:
```
npm run dev
```

The application will be available at http://localhost:3000.

## Project Structure

- `/api` - Python backend for data simulation and analysis
- `/app` - Next.js frontend application
  - `/components` - Reusable UI components
  - `/hooks` - Custom React hooks
  - `/types` - TypeScript interfaces
  - `/utils` - Utility functions
  - `/maintenance` - Maintenance scheduling page

## How It Works

1. The Python backend simulates real-time sensor data from industrial machines
2. Data is analyzed for anomalies and trends using statistical methods
3. Results are streamed to the frontend via WebSockets
4. The frontend visualizes the data and provides maintenance planning tools

## Future Enhancements

- Integration with real sensor data sources
- More sophisticated machine learning models for prediction
- Mobile application for technicians
- Integration with work order systems
- Expanded analytics for efficiency optimization

## License

This project is licensed under the MIT License - see the LICENSE file for details.#   T i c - T e c h - T o e - 2 0 2 5  
 