"""Shared state checks, with no plotting or derived-label generation."""

import json

from pathlib import Path

import numpy as np

CAMERA_NAMES = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")

INITIAL_TOLERANCES={"position_m":.01,"velocity_m_s":.02,"rotation_deg":.1,
                    "angular_velocity_deg_s":.1,"acceleration_m_s2":.1}

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)

def state_errors(states,reference,indices):
    indices=np.asarray(indices,dtype=int)
    rotation=abs((states["rotation_deg"][indices]-reference["rotation_deg"][indices]+180)%360-180)
    result={"rotation_deg":float(rotation.max())}
    for key,out_key in (("position","position_m"),("velocity","velocity_m_s"),
                        ("angular_velocity","angular_velocity_deg_s"),("acceleration","acceleration_m_s2")):
        result[out_key]=float(np.linalg.norm(states[key][indices]-reference[key][indices],axis=-1).max())
    return result
