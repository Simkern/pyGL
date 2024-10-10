import re
import numpy as np

def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = {
        'info': [],
        'rk': [],
        'warn': []
    }
    
    dlra_info_pattern = re.compile(r"DLRA_INFO: Step\s+(\d+), T =\s+([\d.]+): dX = ([\d.E+-]+) X = ([\d.E+-]+) dX/X = ([\d.E+-]+)")
    dlra_rk_pattern   = re.compile(r"RA-DLRA:   Step\s+(\d+), T =\s+([\d.]+): rk = (\d+)")
    dlra_warn_pattern = re.compile(r"RA-DLRA: WARN:")
    
    for line in lines:
        match = dlra_info_pattern.search(line)
        if match:
            step, time, dx, x, dx_x = match.groups()
            data['info'].append({
                'step': int(step),
                'time': float(time),
                'dx': float(dx),
                'x': float(x),
                'dx_x': float(dx_x)
            })        

        match = dlra_rk_pattern.search(line)
        if match:
            step, time, rk = match.groups()
            data['rk'].append({
                'step': int(step),
                'time': float(time),
                'dx': float(dx),
                'x': float(x),
                'dx_x': float(dx_x)
            })
        
        warn_match = dlra_warn_pattern.search(line)
        if warn_match:
            warning = warn_match.group(1).strip()
            data['warn'].append(warning)
    
    return data

# Example usage:
log_file_path = 'logfile_GL_dt_kexpm_n0128_dt_0.10E-01_tol_0.10E-05.txt'
parsed_data = parse_log_file(log_file_path)

# Print extracted data
for step_info in parsed_data['info']:
    print(f"Step {step_info['step']} at time {step_info['time']} has dX = {step_info['dx']} and X = {step_info['x']} with dX/X = {step_info['dx_x']}")

print("\nWarnings:")
for warning in parsed_data['warnings']:
    print(warning)
