"""
Machine configurations for synthetic data generation.
Contains specifications for all five machine types.
"""

# Machine specifications with sensor parameters and normal operating ranges
MACHINE_CONFIGS = {
    # 1. Siemens SIMOTICS Electric Motors
    "siemens_motor": {
        "sensors": {
            "temperature": {"mean": 70, "std_dev": 3, "min_val": 60, "max_val": 95},
            "vibration": {"mean": 3, "std_dev": 0.5, "min_val": 1, "max_val": 15},
            "current": {"mean": 60, "std_dev": 5, "min_val": 10, "max_val": 100},
            "voltage": {"mean": 380, "std_dev": 5, "min_val": 370, "max_val": 400}
        },
        "failure_patterns": {
            # Bearing failure pattern
            "bearing_failure": {
                "affected_sensors": ["vibration", "temperature"],
                "degradation": {
                    "vibration": {"start": 0.65, "end": 0.85, "pattern": "exponential", "severity": 2.0},
                    "temperature": {"start": 0.70, "end": 0.85, "pattern": "linear", "severity": 0.5}
                },
                "failure_point": 0.85
            },
            # Winding failure pattern
            "winding_failure": {
                "affected_sensors": ["temperature", "current"],
                "degradation": {
                    "temperature": {"start": 0.55, "end": 0.80, "pattern": "exponential", "severity": 1.5},
                    "current": {"start": 0.65, "end": 0.80, "pattern": "step", "severity": 0.4}
                },
                "failure_point": 0.80
            }
        },
        "maintenance_interval_days": 45
    },
    
    # 2. ABB Dodge Mounted Bearings
    "abb_bearing": {
        "sensors": {
            "vibration": {"mean": 1.0, "std_dev": 0.2, "min_val": 0.2, "max_val": 5},
            "temperature": {"mean": 50, "std_dev": 2, "min_val": 40, "max_val": 80},
            "acoustic": {"mean": 55, "std_dev": 3, "min_val": 40, "max_val": 85}
        },
        "failure_patterns": {
            # Lubrication failure
            "lubrication_failure": {
                "affected_sensors": ["vibration", "temperature", "acoustic"],
                "degradation": {
                    "vibration": {"start": 0.60, "end": 0.78, "pattern": "exponential", "severity": 3.0},
                    "temperature": {"start": 0.65, "end": 0.78, "pattern": "linear", "severity": 1.0},
                    "acoustic": {"start": 0.55, "end": 0.78, "pattern": "linear", "severity": 1.2}
                },
                "failure_point": 0.78
            },
            # Contamination issue
            "contamination": {
                "affected_sensors": ["vibration", "acoustic"],
                "degradation": {
                    "vibration": {"start": 0.40, "end": 0.90, "pattern": "step", "severity": 1.5},
                    "acoustic": {"start": 0.45, "end": 0.90, "pattern": "linear", "severity": 0.8}
                },
                "failure_point": 0.90
            }
        },
        "maintenance_interval_days": 30
    },
    
    # 3. HAAS VF-2 CNC Milling Machine
    "haas_cnc": {
        "sensors": {
            "spindle_load": {"mean": 40, "std_dev": 5, "min_val": 20, "max_val": 95},
            "vibration": {"mean": 1.5, "std_dev": 0.3, "min_val": 0.5, "max_val": 6},
            "temperature": {"mean": 55, "std_dev": 2, "min_val": 45, "max_val": 85},
            "acoustic": {"mean": 65, "std_dev": 3, "min_val": 50, "max_val": 90}
        },
        "failure_patterns": {
            # Tool wear
            "tool_wear": {
                "affected_sensors": ["spindle_load", "vibration", "acoustic"],
                "degradation": {
                    "spindle_load": {"start": 0.30, "end": 0.75, "pattern": "linear", "severity": 1.2},
                    "vibration": {"start": 0.40, "end": 0.75, "pattern": "exponential", "severity": 1.8},
                    "acoustic": {"start": 0.35, "end": 0.75, "pattern": "linear", "severity": 0.7}
                },
                "failure_point": 0.75
            },
            # Spindle issue
            "spindle_failure": {
                "affected_sensors": ["spindle_load", "temperature", "vibration"],
                "degradation": {
                    "spindle_load": {"start": 0.70, "end": 0.88, "pattern": "step", "severity": 1.5},
                    "temperature": {"start": 0.72, "end": 0.88, "pattern": "exponential", "severity": 1.2},
                    "vibration": {"start": 0.65, "end": 0.88, "pattern": "exponential", "severity": 2.5}
                },
                "failure_point": 0.88
            }
        },
        "maintenance_interval_days": 20
    },
    
    # 4. Grundfos CR Vertical Multistage Pumps
    "grundfos_pump": {
        "sensors": {
            "pressure": {"mean": 15, "std_dev": 1, "min_val": 2, "max_val": 25},
            "flow_rate": {"mean": 90, "std_dev": 8, "min_val": 1, "max_val": 180},
            "temperature": {"mean": 55, "std_dev": 3, "min_val": 40, "max_val": 85},
            "power": {"mean": 10, "std_dev": 1.5, "min_val": 0.37, "max_val": 22}
        },
        "failure_patterns": {
            # Cavitation
            "cavitation": {
                "affected_sensors": ["pressure", "flow_rate", "power"],
                "degradation": {
                    "pressure": {"start": 0.50, "end": 0.82, "pattern": "step", "severity": 0.8},
                    "flow_rate": {"start": 0.55, "end": 0.82, "pattern": "linear", "severity": -0.5},  # Decrease
                    "power": {"start": 0.60, "end": 0.82, "pattern": "exponential", "severity": 1.0}
                },
                "failure_point": 0.82
            },
            # Impeller wear
            "impeller_wear": {
                "affected_sensors": ["flow_rate", "pressure", "power"],
                "degradation": {
                    "flow_rate": {"start": 0.40, "end": 0.90, "pattern": "linear", "severity": -0.7},  # Decrease
                    "pressure": {"start": 0.45, "end": 0.90, "pattern": "linear", "severity": -0.5},  # Decrease
                    "power": {"start": 0.50, "end": 0.90, "pattern": "exponential", "severity": 1.2}
                },
                "failure_point": 0.90
            }
        },
        "maintenance_interval_days": 60
    },
    
    # 5. Carrier 30XA Air-Cooled Chiller (HVAC)
    "carrier_chiller": {
        "sensors": {
            "refrigerant_pressure": {"mean": 15, "std_dev": 1, "min_val": 8, "max_val": 25},
            "condenser_temp": {"mean": 40, "std_dev": 1, "min_val": 35, "max_val": 55},
            "evaporator_temp": {"mean": 7.5, "std_dev": 0.5, "min_val": 5, "max_val": 15},
            "power": {"mean": 0.6, "std_dev": 0.03, "min_val": 0.5, "max_val": 0.7}
        },
        "failure_patterns": {
            # Refrigerant leak
            "refrigerant_leak": {
                "affected_sensors": ["refrigerant_pressure", "evaporator_temp", "power"],
                "degradation": {
                    "refrigerant_pressure": {"start": 0.60, "end": 0.85, "pattern": "linear", "severity": -0.8},  # Decrease
                    "evaporator_temp": {"start": 0.65, "end": 0.85, "pattern": "linear", "severity": 0.7},
                    "power": {"start": 0.70, "end": 0.85, "pattern": "step", "severity": 0.5}
                },
                "failure_point": 0.85
            },
            # Condenser fouling
            "condenser_fouling": {
                "affected_sensors": ["condenser_temp", "refrigerant_pressure", "power"],
                "degradation": {
                    "condenser_temp": {"start": 0.45, "end": 0.92, "pattern": "exponential", "severity": 0.6},
                    "refrigerant_pressure": {"start": 0.50, "end": 0.92, "pattern": "exponential", "severity": 0.9},
                    "power": {"start": 0.55, "end": 0.92, "pattern": "linear", "severity": 0.8}
                },
                "failure_point": 0.92
            }
        },
        "maintenance_interval_days": 90
    }
} 