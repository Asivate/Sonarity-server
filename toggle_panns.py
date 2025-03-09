#!/usr/bin/env python3
"""
Toggle PANNs Model Script for SoundWatch

This script sets the USE_PANNS_MODEL environment variable and restarts the server.
It provides an easy way to switch between the PANNs model and other sound recognition models.
"""

import os
import sys
import subprocess
import logging
import argparse
import platform
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_current_setting():
    """Get the current PANNs model setting from environment variable"""
    current_setting = os.environ.get('USE_PANNS_MODEL', '0')
    return current_setting == '1'

def toggle_setting(use_panns=None):
    """
    Toggle or set the PANNs model setting
    
    Args:
        use_panns: If None, toggle the current setting. If True/False, set to that value.
        
    Returns:
        str: The new setting ('0' or '1')
    """
    if use_panns is None:
        # Toggle the current setting
        current = get_current_setting()
        new_setting = '0' if current else '1'
    else:
        # Set to the specified value
        new_setting = '1' if use_panns else '0'
        
    return new_setting

def restart_server(use_panns):
    """
    Restart the server with the new setting
    
    Args:
        use_panns (str): The new setting ('0' or '1')
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if server.py exists
    server_script = Path('server.py')
    if not server_script.exists():
        # Check if we need to change directory
        parent_dir = Path('server')
        if parent_dir.exists() and (parent_dir / 'server.py').exists():
            os.chdir(parent_dir)
        else:
            logger.error("Cannot find server.py. Make sure you're in the right directory.")
            return False
    
    # Set environment variables for the new process
    env = os.environ.copy()
    env["USE_PANNS_MODEL"] = use_panns
    
    # Disable other models when enabling PANNs (avoid conflicts)
    if use_panns == '1':
        env["USE_AST_MODEL"] = '0'
        logger.info("Setting USE_AST_MODEL=0 (disabled) to avoid conflicts")
    
    logger.info(f"Setting USE_PANNS_MODEL={use_panns}")
    
    # Determine the Python executable
    python_exe = "python" if platform.system() == "Windows" else "python3"
    
    # Build the server command
    cmd = [python_exe, "server.py", "--port=8080"]
    
    try:
        # Start the server as a new process
        logger.info(f"Starting server with command: {' '.join(cmd)}")
        subprocess.Popen(cmd, env=env)
        return True
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Toggle or set the PANNs model for SoundWatch")
    parser.add_argument('setting', nargs='?', choices=['on', 'off', 'enable', 'disable'], 
                        help="Set the PANNs model to 'on'/'enable' or 'off'/'disable', or omit to toggle")
    args = parser.parse_args()
    
    # Determine the new setting based on arguments
    if args.setting:
        use_panns = args.setting in ['on', 'enable']
    else:
        use_panns = None  # Toggle the current setting
    
    # Get the new setting string
    new_setting = toggle_setting(use_panns)
    
    # Show the current and new settings
    current = get_current_setting()
    logger.info(f"Current setting: PANNs model is {'enabled' if current else 'disabled'}")
    logger.info(f"New setting: PANNs model will be {'enabled' if new_setting == '1' else 'disabled'}")
    
    # Ask if the user wants to restart the server
    response = input("Do you want to restart the server with the new setting? (y/n): ")
    if response.lower() in ['y', 'yes']:
        if restart_server(new_setting):
            logger.info("Server restarted successfully with the new setting")
        else:
            logger.error("Failed to restart server")
    else:
        logger.info("Server not restarted. The new setting will apply the next time you start the server.")
        logger.info(f"To manually start the server with this setting, run:")
        if platform.system() == "Windows":
            logger.info(f"$env:USE_PANNS_MODEL={new_setting}")
        else:
            logger.info(f"export USE_PANNS_MODEL={new_setting}")
        logger.info(f"python{'3' if platform.system() != 'Windows' else ''} server.py --port=8080")
    
if __name__ == "__main__":
    main() 