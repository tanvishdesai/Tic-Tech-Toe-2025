"""
Machine-specific data generators for each type of equipment.
These generators extend the base MachineDataGenerator with specific behavior.
"""

import random
from datetime import datetime, timedelta
import numpy as np
from src.data_generation.data_generator import MachineDataGenerator
from src.data_generation.machine_configs import MACHINE_CONFIGS

class SiemensMotorGenerator(MachineDataGenerator):
    """Generator for Siemens SIMOTICS Electric Motors data."""
    
    def __init__(self, machine_id, failure_type=None, **kwargs):
        """
        Initialize Siemens motor data generator.
        
        Args:
            machine_id: Unique identifier for the machine
            failure_type: Type of failure to simulate (bearing_failure, winding_failure, or None)
            **kwargs: Additional arguments passed to MachineDataGenerator
        """
        super().__init__(machine_id, **kwargs)
        self.config = MACHINE_CONFIGS["siemens_motor"]
        self.failure_type = failure_type
        
        # Add base sensor data
        for sensor_name, params in self.config["sensors"].items():
            self.add_normal_sensor_reading(
                sensor_name=sensor_name,
                mean=params["mean"],
                std_dev=params["std_dev"],
                min_val=params["min_val"],
                max_val=params["max_val"]
            )
        
        # Add daily load cycle (higher during day, lower at night)
        self.add_cyclical_pattern("current", amplitude=0.15, period_hours=24)
        
        # Add maintenance events
        self.add_maintenance_events(
            maintenance_interval_days=self.config["maintenance_interval_days"]
        )
        
        # Add failure pattern if specified
        if failure_type and failure_type in self.config["failure_patterns"]:
            failure_pattern = self.config["failure_patterns"][failure_type]
            for sensor in failure_pattern["affected_sensors"]:
                degradation = failure_pattern["degradation"][sensor]
                self.add_degradation_pattern(
                    sensor_name=sensor,
                    start_percent=degradation["start"],
                    end_percent=degradation["end"],
                    pattern_type=degradation["pattern"],
                    severity=degradation["severity"]
                )
            
            # Add sudden failure event
            sensor = random.choice(failure_pattern["affected_sensors"])
            self.add_sudden_failure(sensor, failure_pattern["failure_point"], duration_hours=3)
        
        # Add anomaly and failure labels
        self.add_anomaly_labels()
        self.add_failure_labels()


class AbbBearingGenerator(MachineDataGenerator):
    """Generator for ABB Dodge Mounted Bearings data."""
    
    def __init__(self, machine_id, failure_type=None, **kwargs):
        """
        Initialize ABB bearing data generator.
        
        Args:
            machine_id: Unique identifier for the machine
            failure_type: Type of failure to simulate (lubrication_failure, contamination, or None)
            **kwargs: Additional arguments passed to MachineDataGenerator
        """
        super().__init__(machine_id, **kwargs)
        self.config = MACHINE_CONFIGS["abb_bearing"]
        self.failure_type = failure_type
        
        # Add base sensor data
        for sensor_name, params in self.config["sensors"].items():
            self.add_normal_sensor_reading(
                sensor_name=sensor_name,
                mean=params["mean"],
                std_dev=params["std_dev"],
                min_val=params["min_val"],
                max_val=params["max_val"]
            )
        
        # Add minor temperature cycle based on ambient temperature
        self.add_cyclical_pattern("temperature", amplitude=0.1, period_hours=24)
        
        # Add maintenance events
        self.add_maintenance_events(
            maintenance_interval_days=self.config["maintenance_interval_days"]
        )
        
        # Add failure pattern if specified
        if failure_type and failure_type in self.config["failure_patterns"]:
            failure_pattern = self.config["failure_patterns"][failure_type]
            for sensor in failure_pattern["affected_sensors"]:
                degradation = failure_pattern["degradation"][sensor]
                self.add_degradation_pattern(
                    sensor_name=sensor,
                    start_percent=degradation["start"],
                    end_percent=degradation["end"],
                    pattern_type=degradation["pattern"],
                    severity=degradation["severity"]
                )
            
            # Add sudden failure event
            sensor = random.choice(failure_pattern["affected_sensors"])
            self.add_sudden_failure(sensor, failure_pattern["failure_point"], duration_hours=2)
        
        # Add anomaly and failure labels
        self.add_anomaly_labels()
        self.add_failure_labels()


class HaasCncGenerator(MachineDataGenerator):
    """Generator for HAAS VF-2 CNC Milling Machine data."""
    
    def __init__(self, machine_id, failure_type=None, **kwargs):
        """
        Initialize HAAS CNC data generator.
        
        Args:
            machine_id: Unique identifier for the machine
            failure_type: Type of failure to simulate (tool_wear, spindle_failure, or None)
            **kwargs: Additional arguments passed to MachineDataGenerator
        """
        super().__init__(machine_id, **kwargs)
        self.config = MACHINE_CONFIGS["haas_cnc"]
        self.failure_type = failure_type
        
        # Add base sensor data
        for sensor_name, params in self.config["sensors"].items():
            self.add_normal_sensor_reading(
                sensor_name=sensor_name,
                mean=params["mean"],
                std_dev=params["std_dev"],
                min_val=params["min_val"],
                max_val=params["max_val"]
            )
        
        # Add work shift pattern (higher load during work hours)
        self.add_cyclical_pattern("spindle_load", amplitude=0.25, period_hours=8)
        
        # Add maintenance events
        self.add_maintenance_events(
            maintenance_interval_days=self.config["maintenance_interval_days"]
        )
        
        # Add failure pattern if specified
        if failure_type and failure_type in self.config["failure_patterns"]:
            failure_pattern = self.config["failure_patterns"][failure_type]
            for sensor in failure_pattern["affected_sensors"]:
                degradation = failure_pattern["degradation"][sensor]
                self.add_degradation_pattern(
                    sensor_name=sensor,
                    start_percent=degradation["start"],
                    end_percent=degradation["end"],
                    pattern_type=degradation["pattern"],
                    severity=degradation["severity"]
                )
            
            # Add sudden failure event
            sensor = random.choice(failure_pattern["affected_sensors"])
            self.add_sudden_failure(sensor, failure_pattern["failure_point"], duration_hours=1)
        
        # Add anomaly and failure labels
        self.add_anomaly_labels()
        self.add_failure_labels()


class GrundfosPumpGenerator(MachineDataGenerator):
    """Generator for Grundfos CR Vertical Multistage Pumps data."""
    
    def __init__(self, machine_id, failure_type=None, **kwargs):
        """
        Initialize Grundfos pump data generator.
        
        Args:
            machine_id: Unique identifier for the machine
            failure_type: Type of failure to simulate (cavitation, impeller_wear, or None)
            **kwargs: Additional arguments passed to MachineDataGenerator
        """
        super().__init__(machine_id, **kwargs)
        self.config = MACHINE_CONFIGS["grundfos_pump"]
        self.failure_type = failure_type
        
        # Add base sensor data
        for sensor_name, params in self.config["sensors"].items():
            self.add_normal_sensor_reading(
                sensor_name=sensor_name,
                mean=params["mean"],
                std_dev=params["std_dev"],
                min_val=params["min_val"],
                max_val=params["max_val"]
            )
        
        # Add demand fluctuation pattern
        self.add_cyclical_pattern("flow_rate", amplitude=0.2, period_hours=12)
        self.add_cyclical_pattern("power", amplitude=0.15, period_hours=12)
        
        # Add maintenance events
        self.add_maintenance_events(
            maintenance_interval_days=self.config["maintenance_interval_days"]
        )
        
        # Add failure pattern if specified
        if failure_type and failure_type in self.config["failure_patterns"]:
            failure_pattern = self.config["failure_patterns"][failure_type]
            for sensor in failure_pattern["affected_sensors"]:
                degradation = failure_pattern["degradation"][sensor]
                self.add_degradation_pattern(
                    sensor_name=sensor,
                    start_percent=degradation["start"],
                    end_percent=degradation["end"],
                    pattern_type=degradation["pattern"],
                    severity=degradation["severity"]
                )
            
            # Add sudden failure event
            sensor = random.choice(failure_pattern["affected_sensors"])
            self.add_sudden_failure(sensor, failure_pattern["failure_point"], duration_hours=5)
        
        # Add anomaly and failure labels
        self.add_anomaly_labels()
        self.add_failure_labels()


class CarrierChillerGenerator(MachineDataGenerator):
    """Generator for Carrier 30XA Air-Cooled Chiller (HVAC) data."""
    
    def __init__(self, machine_id, failure_type=None, **kwargs):
        """
        Initialize Carrier chiller data generator.
        
        Args:
            machine_id: Unique identifier for the machine
            failure_type: Type of failure to simulate (refrigerant_leak, condenser_fouling, or None)
            **kwargs: Additional arguments passed to MachineDataGenerator
        """
        super().__init__(machine_id, **kwargs)
        self.config = MACHINE_CONFIGS["carrier_chiller"]
        self.failure_type = failure_type
        
        # Add base sensor data
        for sensor_name, params in self.config["sensors"].items():
            self.add_normal_sensor_reading(
                sensor_name=sensor_name,
                mean=params["mean"],
                std_dev=params["std_dev"],
                min_val=params["min_val"],
                max_val=params["max_val"]
            )
        
        # Add outdoor temperature effect on condenser temp
        self.add_cyclical_pattern("condenser_temp", amplitude=0.2, period_hours=24)
        
        # Add load pattern (higher during day)
        self.add_cyclical_pattern("power", amplitude=0.2, period_hours=24)
        
        # Add maintenance events
        self.add_maintenance_events(
            maintenance_interval_days=self.config["maintenance_interval_days"]
        )
        
        # Add failure pattern if specified
        if failure_type and failure_type in self.config["failure_patterns"]:
            failure_pattern = self.config["failure_patterns"][failure_type]
            for sensor in failure_pattern["affected_sensors"]:
                degradation = failure_pattern["degradation"][sensor]
                self.add_degradation_pattern(
                    sensor_name=sensor,
                    start_percent=degradation["start"],
                    end_percent=degradation["end"],
                    pattern_type=degradation["pattern"],
                    severity=degradation["severity"]
                )
            
            # Add sudden failure event
            sensor = random.choice(failure_pattern["affected_sensors"])
            self.add_sudden_failure(sensor, failure_pattern["failure_point"], duration_hours=8)
        
        # Add anomaly and failure labels
        self.add_anomaly_labels()
        self.add_failure_labels()


# Factory function to create the right generator based on machine type
def create_machine_generator(machine_type, machine_id, failure_type=None, **kwargs):
    """
    Factory function to create the appropriate machine generator.
    
    Args:
        machine_type: Type of machine ('siemens_motor', 'abb_bearing', etc.)
        machine_id: Unique identifier for the machine
        failure_type: Type of failure to simulate (specific to machine type)
        **kwargs: Additional arguments passed to the generator
        
    Returns:
        MachineDataGenerator: The appropriate generator instance
    """
    generators = {
        "siemens_motor": SiemensMotorGenerator,
        "abb_bearing": AbbBearingGenerator,
        "haas_cnc": HaasCncGenerator,
        "grundfos_pump": GrundfosPumpGenerator,
        "carrier_chiller": CarrierChillerGenerator
    }
    
    if machine_type not in generators:
        raise ValueError(f"Unknown machine type: {machine_type}")
    
    return generators[machine_type](machine_id, failure_type, **kwargs) 