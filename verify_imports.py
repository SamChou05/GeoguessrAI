#!/usr/bin/env python3
"""
Verification script to check if the code refactoring is working correctly.
This script checks imports, class definitions, and basic functionality.
"""

import sys
import os

def check_imports():
    """Check if all imports work correctly"""
    print("=" * 60)
    print("Checking imports...")
    print("=" * 60)
    
    try:
        # Check if demo.py can be imported (this will fail if dependencies missing)
        from demo import GeoguessrAI
        print("✓ Successfully imported GeoguessrAI from demo.py")
        return True
    except ImportError as e:
        print(f"⚠ Import failed (likely missing dependencies): {e}")
        print("  This is expected if PyTorch/torchvision are not installed.")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def check_class_structure():
    """Check class structure without importing"""
    print("\n" + "=" * 60)
    print("Checking class structure...")
    print("=" * 60)
    
    with open('demo.py', 'r') as f:
        content = f.read()
        
    checks = [
        ('class GeoguessrAI', 'Class definition'),
        ('def __init__(self)', 'Initialization method'),
        ('def guess(self, image_path)', 'Guess method'),
        ('self.gm', 'Gaussian mixture model reference'),
        ('self.model', 'Model reference'),
    ]
    
    all_passed = True
    for check, desc in checks:
        if check in content:
            print(f"✓ {desc} found")
        else:
            print(f"✗ {desc} NOT found")
            all_passed = False
    
    return all_passed

def check_test_file():
    """Check test.py references"""
    print("\n" + "=" * 60)
    print("Checking test.py...")
    print("=" * 60)
    
    with open('test.py', 'r') as f:
        content = f.read()
    
    checks = [
        ('from demo import GeoguessrAI', 'Import statement'),
        ('geoguessr_ai = GeoguessrAI()', 'Class instantiation'),
        ('geoguessr_ai.guess', 'Method calls'),
    ]
    
    all_passed = True
    for check, desc in checks:
        if check in content:
            print(f"✓ {desc} found")
        else:
            print(f"✗ {desc} NOT found")
            all_passed = False
    
    # Check for old references
    if 'GeoKnowr' in content:
        print("✗ WARNING: Found old 'GeoKnowr' reference")
        all_passed = False
    else:
        print("✓ No old 'GeoKnowr' references")
    
    if 'geo_knowr' in content:
        print("✗ WARNING: Found old 'geo_knowr' variable")
        all_passed = False
    else:
        print("✓ No old 'geo_knowr' variables")
    
    return all_passed

def check_model_files():
    """Check if model files exist"""
    print("\n" + "=" * 60)
    print("Checking model files...")
    print("=" * 60)
    
    model_files = [
        'model/gm.pkl',
        'model/resnet.pt'
    ]
    
    all_exist = True
    for file in model_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} NOT found")
            all_exist = False
    
    return all_exist

def check_demo_images():
    """Check if demo images exist"""
    print("\n" + "=" * 60)
    print("Checking demo images...")
    print("=" * 60)
    
    demo_dir = 'demo_in'
    if os.path.exists(demo_dir):
        images = [f for f in os.listdir(demo_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"✓ Found {len(images)} images in {demo_dir}/")
        return len(images) > 0
    else:
        print(f"✗ {demo_dir}/ directory not found")
        return False

def try_instantiate():
    """Try to instantiate the class (will fail if dependencies missing)"""
    print("\n" + "=" * 60)
    print("Attempting to instantiate GeoguessrAI...")
    print("=" * 60)
    
    try:
        from demo import GeoguessrAI
        ai = GeoguessrAI()
        print("✓ Successfully instantiated GeoguessrAI")
        print(f"  - Number of classes: {ai.num_classes}")
        return True
    except FileNotFoundError as e:
        print(f"⚠ Model files not found: {e}")
        return False
    except ImportError as e:
        print(f"⚠ Missing dependencies: {e}")
        print("  Install with: pip install torch torchvision scikit-learn")
        return False
    except Exception as e:
        print(f"⚠ Error during instantiation: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("GeoguessrAI Code Verification")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run checks
    results.append(("Class Structure", check_class_structure()))
    results.append(("Test File", check_test_file()))
    results.append(("Model Files", check_model_files()))
    results.append(("Demo Images", check_demo_images()))
    
    # Try imports (may fail if deps missing)
    import_success = check_imports()
    if import_success:
        results.append(("Class Instantiation", try_instantiate()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for check_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All structural checks passed!")
        print("  The code refactoring appears to be correct.")
        if not import_success:
            print("\n  Note: Dependencies may need to be installed to run the code.")
    else:
        print("\n⚠ Some checks failed. Review the output above.")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())

