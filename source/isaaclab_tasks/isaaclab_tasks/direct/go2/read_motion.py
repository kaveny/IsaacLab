"""
Module for reading motion data from files and converting between different robot formats.
Specifically handles conversion from A1 robot motion data to Go2 format.
"""

import json
import numpy as np

def read_frames_from_json(file_path) -> np.array:
    """
    Read motion frames from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing frames
        
    Returns:
        A list of frames or the entire frames data structure
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        frames = data.get('Frames', data)  # fallback to entire data if no 'frames' key
        
        print(f"Successfully read {len(frames) if isinstance(frames, list) else 'data'} from {file_path}")
        return np.array(frames)
    
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None
    except Exception as e:
        print(f"Error reading frames: {e}")
        return None

def convert_a1_to_go2_format(frames_a1: np.array) -> np.array:
    """
    Convert frames from A1 robot format to go2 robot format.
    
    The main difference is in the joint angle ordering:
    - A1:  [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
    - go2: [FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf]
    
    Args:
        frames_a1: Numpy array of frames in A1 format
        
    Returns:
        Numpy array of frames in go2 format
    """
    if frames_a1 is None:
        return None
    
    # Create a new array with the same shape as the input
    frames_go2 = np.copy(frames_a1)
    
    for i in range(len(frames_a1)):
        # The first 7 values (base position and orientation) remain the same
        # Only remap the 12 joint angles (indices 7-19)
        
        # Extract joint angles from A1 format
        a1_joints = frames_a1[i, 7:19]
        
        # Map to go2 format:
        # A1: [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
        # go2: [FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf]
        
        go2_joints = np.zeros(12)
        
        # Hip joints
        go2_joints[0] = a1_joints[3]  # FL_hip
        go2_joints[1] = a1_joints[0]  # FR_hip
        go2_joints[2] = a1_joints[9]  # RL_hip
        go2_joints[3] = a1_joints[6]  # RR_hip
        
        # Thigh joints
        go2_joints[4] = a1_joints[4]  # FL_thigh
        go2_joints[5] = a1_joints[1]  # FR_thigh
        go2_joints[6] = a1_joints[10]  # RL_thigh
        go2_joints[7] = a1_joints[7]  # RR_thigh
        
        # Calf joints
        go2_joints[8] = a1_joints[5]   # FL_calf
        go2_joints[9] = a1_joints[2]   # FR_calf
        go2_joints[10] = a1_joints[11]  # RL_calf
        go2_joints[11] = a1_joints[8]   # RR_calf
        
        # Replace the joint angles in the output frame
        frames_go2[i, 7:19] = go2_joints
    
    return frames_go2

def get_next_front_right_joint_position(frames_go2, current_index=0):
    """
    Get the next joint position for the front right leg (hip, thigh, calf) from frames data.
    When the end of the array is reached, it loops back to the first row.
    
    Args:
        frames_go2: Numpy array of frames in go2 format
        current_index: Current index in the frames array (defaults to 0)
        
    Returns:
        A tuple of (hip, thigh, calf) joint positions and the next index
    """
    if frames_go2 is None or len(frames_go2) == 0:
        return None, current_index
    
    # Calculate the next index, wrapping around if needed
    next_index = (current_index + 1) % len(frames_go2)
    
    # Extract front right joint positions from go2 format
    # FR_hip is at index 1, FR_thigh at index 5, FR_calf at index 9
    fr_hip = frames_go2[current_index, 7 + 1]   # FR_hip
    fr_thigh = frames_go2[current_index, 7 + 5] # FR_thigh
    fr_calf = frames_go2[current_index, 7 + 9]  # FR_calf
    
    return (fr_hip, fr_thigh, fr_calf), next_index


'''example usage'''
# file = 'trot_A1.txt'
# f = read_frames_from_json(file)
# go2_f = convert_a1_to_go2_format(f)
# joints, idx = get_next_front_right_joint_position(go2_f, (idx+1)% len(go2_f))