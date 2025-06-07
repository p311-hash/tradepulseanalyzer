"""Check if all required dependencies are installed."""
import sys

def check_dependency(module_name):
    try:
        __import__(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError:
        print(f"✗ {module_name}")
        return False

required = [
    'torch',
    'numpy',
    'pandas',
    'sklearn',
    'talib',
    'joblib'
]

print("Checking dependencies...")
all_good = all(check_dependency(dep) for dep in required)
sys.exit(0 if all_good else 1)
