#!/usr/bin/env python3
"""
Test runner with comprehensive logging and error handling.
Captures all output to log files for debugging when tests fail or crash.
"""

import os
import sys
import traceback
import logging
import subprocess
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
import io

def setup_logging(test_name):
    """Setup comprehensive logging for test runs"""
    # Create logs directory
    log_dir = os.path.join(os.path.dirname(__file__), 'test_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup file handlers
    log_file = os.path.join(log_dir, f"{test_name}_{timestamp}.log")
    error_file = os.path.join(log_dir, f"{test_name}_{timestamp}_errors.log")
    
    # Configure main logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)  # Also print to console
        ]
    )
    
    # Configure error logger
    error_logger = logging.getLogger('errors')
    error_handler = logging.FileHandler(error_file, mode='w')
    error_handler.setFormatter(logging.Formatter('%(asctime)s - ERROR - %(message)s'))
    error_logger.addHandler(error_handler)
    error_logger.setLevel(logging.ERROR)
    
    return logging.getLogger(), error_logger, log_file, error_file

def run_test_safely(test_script, test_name):
    """Run a test script with comprehensive error handling and logging"""
    logger, error_logger, log_file, error_file = setup_logging(test_name)
    
    logger.info(f"="*60)
    logger.info(f"Starting {test_name} test")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Error file: {error_file}")
    logger.info(f"="*60)
    
    try:
        # Change to the test directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(test_dir)
        logger.info(f"Working directory: {test_dir}")
        
        # Run test using subprocess to properly handle Hydra configs
        logger.info(f"Running subprocess: python {test_script}")
        
        result = subprocess.run(
            [sys.executable, test_script], 
            capture_output=True, 
            text=True, 
            timeout=300,
            cwd=test_dir
        )
        
        stdout_content = result.stdout
        stderr_content = result.stderr
        
        # Log captured output
        if stdout_content:
            logger.info("STDOUT OUTPUT:")
            logger.info(stdout_content)
            
        if stderr_content:
            logger.error("STDERR OUTPUT:")
            logger.error(stderr_content)
            error_logger.error(stderr_content)
        
        if result.returncode != 0:
            error_msg = f"Test failed with return code {result.returncode}"
            logger.error(error_msg)
            raise subprocess.CalledProcessError(result.returncode, test_script)
        
        logger.info(f"✅ {test_name} completed successfully!")
        return True
        
    except KeyboardInterrupt:
        error_msg = f"❌ {test_name} interrupted by user (Ctrl+C)"
        logger.error(error_msg)
        error_logger.error(error_msg)
        return False
        
    except subprocess.TimeoutExpired:
        error_msg = f"❌ {test_name} timed out (>300 seconds)"
        logger.error(error_msg)
        error_logger.error(error_msg)
        return False
        
    except ImportError as e:
        error_msg = f"❌ {test_name} import error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        error_logger.error(error_msg)
        return False
        
    except Exception as e:
        error_msg = f"❌ {test_name} failed with error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        error_logger.error(error_msg)
        
        # Additional system info for debugging
        try:
            import platform
            import psutil
            
            system_info = f"""
System Information:
- Python: {sys.version}
- Platform: {platform.platform()}
- CPU Count: {psutil.cpu_count()}
- Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB
- Working Directory: {os.getcwd()}
- Python Path: {sys.path[:3]}...
"""
            logger.error(system_info)
            error_logger.error(system_info)
        except ImportError:
            logger.error("Could not gather system info (psutil not available)")
        
        return False
    
    finally:
        logger.info(f"="*60)
        logger.info(f"Test {test_name} finished")
        logger.info(f"Logs saved to: {log_file}")
        if os.path.exists(error_file) and os.path.getsize(error_file) > 0:
            logger.info(f"Errors saved to: {error_file}")
        logger.info(f"="*60)

def main():
    """Run all tests with logging"""
    print("🧪 Running LMGame tests with comprehensive logging...")
    
    tests = [
        # ("ctx_manager_test.py", "ContextManager"),
        # ("es_manager_test.py", "ESManager"), 
        ("agent_proxy_test.py", "AgentProxy"),
    ]
    
    results = {}
    
    for test_script, test_name in tests:
        if os.path.exists(test_script):
            print(f"\n🔄 Running {test_name} test...")
            success = run_test_safely(test_script, test_name)
            results[test_name] = success
        else:
            print(f"⚠️  Test file {test_script} not found, skipping...")
            results[test_name] = False
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if not success:
            all_passed = False
    
    print(f"{'='*60}")
    if all_passed:
        print("🎉 All tests completed successfully!")
    else:
        print("⚠️  Some tests failed. Check log files for details.")
    
    # Show log directory
    log_dir = os.path.join(os.path.dirname(__file__), 'test_logs')
    if os.path.exists(log_dir):
        print(f"📁 Log files saved in: {log_dir}")
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        if log_files:
            print("Recent log files:")
            for log_file in sorted(log_files)[-4:]:  # Show last 4 files
                print(f"  - {log_file}")

if __name__ == "__main__":
    main() 