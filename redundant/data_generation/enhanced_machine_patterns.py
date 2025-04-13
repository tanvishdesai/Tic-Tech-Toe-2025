"""
Enhanced failure patterns and sensor behaviors for improved data generation.
This module provides functions to adjust machine configurations for more distinctive patterns.
"""

import copy
import random
import numpy as np
from src.data_generation.machine_configs import MACHINE_CONFIGS

def get_enhanced_machine_configs():
    """
    Create enhanced versions of the machine configurations with more distinctive patterns.
    
    Returns:
        dict: Enhanced machine configurations
    """
    # Start with a deep copy of the original configs
    enhanced_configs = copy.deepcopy(MACHINE_CONFIGS)
    
    # Enhance each machine type
    for machine_type, config in enhanced_configs.items():
        # 1. Make post-maintenance state more distinctive
        # Increase the impact of maintenance by adding a clear recovery pattern
        config["post_maintenance_recovery"] = {
            "duration_hours": random.randint(12, 48),
            "improvement_factor": random.uniform(0.6, 0.8),
            "recovery_pattern": random.choice(["exponential", "linear"])
        }
        
        # 2. Enhance failure patterns to be more distinctive
        for failure_type, pattern in config["failure_patterns"].items():
            # Make failure points more distinct
            pattern["failure_point"] = min(0.95, pattern["failure_point"] + random.uniform(0.02, 0.05))
            
            # Make degradation patterns start earlier and be more pronounced
            for sensor, degradation in pattern["degradation"].items():
                # Adjust start point to begin earlier
                degradation["start"] = max(0.2, degradation["start"] - random.uniform(0.05, 0.15))
                
                # Increase severity for more clear patterns
                degradation["severity"] = degradation["severity"] * random.uniform(1.2, 1.5)
                
                # Introduce multi-stage degradation for some patterns
                if random.random() > 0.7:
                    degradation["multi_stage"] = True
                    degradation["stage_points"] = sorted([
                        degradation["start"],
                        degradation["start"] + (degradation["end"] - degradation["start"]) * 0.4,
                        degradation["end"]
                    ])
                    degradation["stage_severity"] = [
                        degradation["severity"] * 0.3,
                        degradation["severity"] * 0.7,
                        degradation["severity"]
                    ]
        
        # 3. Add high-frequency pattern changes for NORMAL state 
        # (to make it more distinctive from degrading or post-maintenance)
        config["normal_variation"] = {
            "frequency_hours": random.randint(4, 12),
            "amplitude_factor": random.uniform(0.05, 0.15),
            "sensors": random.sample(list(config["sensors"].keys()), 
                                    k=min(3, len(config["sensors"])))
        }
    
    # Add more distinctive patterns for specific machine types
    
    # Siemens motor - make winding failure more distinctive
    if "siemens_motor" in enhanced_configs:
        motor_config = enhanced_configs["siemens_motor"]
        winding_pattern = motor_config["failure_patterns"].get("winding_failure")
        if winding_pattern:
            # More pronounced temperature increase
            temp_deg = winding_pattern["degradation"]["temperature"]
            temp_deg["severity"] = 2.0
            temp_deg["pattern"] = "exponential" 
            
            # More clear current fluctuations
            current_deg = winding_pattern["degradation"]["current"]
            current_deg["pattern"] = "multi_step"
            current_deg["step_count"] = 4
            current_deg["severity"] = 0.6
            
            # Add a distinctive signature for this failure type
            winding_pattern["signature"] = {
                "current_temperature_ratio": {
                    "start": 0.6,
                    "end": 0.95, 
                    "pattern": "step",
                    "severity": 0.5
                }
            }
    
    # Grundfos pump - make cavitation more distinctive 
    if "grundfos_pump" in enhanced_configs:
        pump_config = enhanced_configs["grundfos_pump"]
        cavitation = pump_config["failure_patterns"].get("cavitation")
        if cavitation:
            # More pronounced pressure fluctuations with unique pattern
            pressure_deg = cavitation["degradation"]["pressure"]
            pressure_deg["pattern"] = "fluctuating"
            pressure_deg["fluctuation_increase"] = 0.1  # Increasing fluctuations as failure approaches
            pressure_deg["severity"] = 1.5
            
            # Add unique sensor interactions for cavitation
            cavitation["sensor_interactions"] = {
                "pressure_flow_correlation": -0.8,  # Strong negative correlation during cavitation
                "interaction_start": 0.5
            }
    
    # Add seasonality effects to make normal vs abnormal more distinct
    for machine_type, config in enhanced_configs.items():
        config["seasonality"] = {
            "daily": {
                "period_hours": 24,
                "sensors": {},
                "amplitude_factor": {}
            },
            "weekly": {
                "period_hours": 168,  # 7 days
                "sensors": {},
                "amplitude_factor": {}
            }
        }
        
        # Add machine-specific seasonality
        for sensor in config["sensors"]:
            # Some sensors have daily patterns
            if random.random() > 0.5:
                config["seasonality"]["daily"]["sensors"][sensor] = True
                config["seasonality"]["daily"]["amplitude_factor"][sensor] = random.uniform(0.05, 0.15)
            
            # Some sensors have weekly patterns
            if random.random() > 0.7:
                config["seasonality"]["weekly"]["sensors"][sensor] = True
                config["seasonality"]["weekly"]["amplitude_factor"][sensor] = random.uniform(0.08, 0.2)
    
    return enhanced_configs

def apply_enhanced_patterns(generator, config, failure_type=None):
    """
    Apply enhanced patterns to a machine data generator.
    
    Args:
        generator: MachineDataGenerator instance
        config: Enhanced machine configuration
        failure_type: Type of failure to simulate (if any)
    
    Returns:
        MachineDataGenerator with enhanced patterns applied
    """
    # Apply seasonality for more distinct normal patterns
    for sensor, params in config["sensors"].items():
        # Add daily seasonality
        if config["seasonality"]["daily"]["sensors"].get(sensor):
            amplitude = params["mean"] * config["seasonality"]["daily"]["amplitude_factor"].get(sensor, 0.1)
            generator.add_cyclical_pattern(sensor, amplitude=amplitude, period_hours=24)
        
        # Add weekly seasonality
        if config["seasonality"]["weekly"]["sensors"].get(sensor):
            amplitude = params["mean"] * config["seasonality"]["weekly"]["amplitude_factor"].get(sensor, 0.15)
            generator.add_cyclical_pattern(sensor, amplitude=amplitude, period_hours=168)
    
    # Add normal operation variations to help distinguish normal state
    if "normal_variation" in config:
        for sensor in config["normal_variation"]["sensors"]:
            # Apply short-term variations that characterize normal operation
            # This makes normal operation more distinct from slow degradation
            mean_val = config["sensors"][sensor]["mean"]
            variation = mean_val * config["normal_variation"]["amplitude_factor"]
            generator.add_cyclical_pattern(
                sensor, 
                amplitude=variation,
                period_hours=config["normal_variation"]["frequency_hours"]
            )
    
    # If failure type provided, apply enhanced failure patterns
    if failure_type and failure_type in config["failure_patterns"]:
        pattern = config["failure_patterns"][failure_type]
        
        # Apply multi-stage degradation if configured
        for sensor in pattern["affected_sensors"]:
            if sensor in pattern["degradation"]:
                degradation = pattern["degradation"][sensor]
                
                if degradation.get("multi_stage"):
                    # Clear any existing degradation (will be replaced with multi-stage)
                    # This isn't ideal in real code, but for this example it's a simple way
                    # to replace default degradation with enhanced version
                    
                    # Apply each stage of degradation
                    for i in range(len(degradation["stage_points"]) - 1):
                        start = degradation["stage_points"][i]
                        end = degradation["stage_points"][i+1]
                        severity = degradation["stage_severity"][i]
                        
                        generator.add_degradation_pattern(
                            sensor_name=sensor,
                            start_percent=start,
                            end_percent=end,
                            pattern_type=degradation["pattern"],
                            severity=severity
                        )
                
                # Apply fluctuations if configured (common in real pump cavitation)
                if degradation.get("pattern") == "fluctuating":
                    # This would require additional implementation in the generator
                    # For this example, we'll approximate it with a higher severity degradation
                    generator.add_degradation_pattern(
                        sensor_name=sensor,
                        start_percent=degradation["start"],
                        end_percent=degradation["end"],
                        pattern_type="exponential",
                        severity=degradation["severity"] * 1.3
                    )
        
        # Apply sensor interactions if configured
        if "sensor_interactions" in pattern:
            # This would require advanced implementation beyond the current generator
            # In a real implementation, we would modify multiple sensors together
            # For this example, we'll just make the failure more distinctive
            
            # Add a sudden failure with higher magnitude
            sensor = random.choice(pattern["affected_sensors"])
            generator.add_sudden_failure(
                sensor_name=sensor,
                failure_start_percent=pattern["failure_point"] - 0.02,
                duration_hours=5,
                magnitude=7.0  # Higher magnitude for more distinctive failure
            )
    
    # Make post-maintenance periods more distinctive
    if "post_maintenance_recovery" in config:
        recovery = config["post_maintenance_recovery"]
        
        # Enhance maintenance effect to make post-maintenance more distinctive
        generator.add_maintenance_events(
            maintenance_interval_days=config.get("maintenance_interval_days", 30),
            maintenance_duration_hours=random.randint(6, 12),
            maintenance_effect=recovery["improvement_factor"]
        )
    
    return generator

# Function to integrate with existing data generation pipeline
def enhance_machine_generator(generator, machine_type, failure_type=None):
    """
    Enhance a machine generator with more distinctive patterns.
    
    Args:
        generator: MachineDataGenerator instance
        machine_type: Type of machine
        failure_type: Type of failure to simulate (if any)
        
    Returns:
        MachineDataGenerator with enhanced patterns
    """
    # Get enhanced configurations
    enhanced_configs = get_enhanced_machine_configs()
    
    if machine_type in enhanced_configs:
        # Apply enhanced patterns
        generator = apply_enhanced_patterns(
            generator, 
            enhanced_configs[machine_type], 
            failure_type
        )
    
    return generator 