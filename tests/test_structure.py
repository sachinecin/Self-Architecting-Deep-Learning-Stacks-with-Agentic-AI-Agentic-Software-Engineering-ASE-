"""
Structure validation tests for KineticStack (no PyTorch required)
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_package_structure():
    """Test that the package structure is correct."""
    print("Testing package structure...")
    
    try:
        import kineticstack
        assert hasattr(kineticstack, '__version__'), "Should have version"
        assert hasattr(kineticstack, '__all__'), "Should have __all__"
        
        expected_exports = ['KineticRuntime', 'SculptorAgent', 'KernelSynthesizerAgent', 'InvariantGuard']
        for export in expected_exports:
            assert export in kineticstack.__all__, f"Should export {export}"
        
        print(f"✓ Package structure valid (version: {kineticstack.__version__})")
        return True
    except Exception as e:
        print(f"✗ Package structure test failed: {e}")
        return False


def test_module_imports():
    """Test that all modules can be imported."""
    print("\nTesting module imports...")
    
    modules_to_test = [
        'kineticstack.core.autograd_sculpting',
        'kineticstack.core.kernel_synthesis',
        'kineticstack.core.runtime',
        'kineticstack.agents.sculptor',
        'kineticstack.agents.kernel_synthesizer',
        'kineticstack.agents.invariant_guard',
    ]
    
    try:
        for module_name in modules_to_test:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Module import test failed: {e}")
        return False


def test_class_definitions():
    """Test that all expected classes are defined."""
    print("\nTesting class definitions...")
    
    try:
        from kineticstack.core.autograd_sculpting import AutogradSculptor, MemoryProfile
        from kineticstack.core.kernel_synthesis import KernelSynthesizer, HardwareProfile, KernelSpec
        from kineticstack.core.runtime import KineticRuntime, RuntimeConfig, OptimizationResult
        from kineticstack.agents.sculptor import SculptorAgent, CheckpointStrategy
        from kineticstack.agents.kernel_synthesizer import KernelSynthesizerAgent, CompilationResult
        from kineticstack.agents.invariant_guard import InvariantGuard, InvariantViolation, ViolationSeverity
        
        classes = [
            ('AutogradSculptor', AutogradSculptor),
            ('MemoryProfile', MemoryProfile),
            ('KernelSynthesizer', KernelSynthesizer),
            ('HardwareProfile', HardwareProfile),
            ('KernelSpec', KernelSpec),
            ('KineticRuntime', KineticRuntime),
            ('RuntimeConfig', RuntimeConfig),
            ('OptimizationResult', OptimizationResult),
            ('SculptorAgent', SculptorAgent),
            ('CheckpointStrategy', CheckpointStrategy),
            ('KernelSynthesizerAgent', KernelSynthesizerAgent),
            ('CompilationResult', CompilationResult),
            ('InvariantGuard', InvariantGuard),
            ('InvariantViolation', InvariantViolation),
            ('ViolationSeverity', ViolationSeverity),
        ]
        
        for name, cls in classes:
            assert cls is not None, f"{name} should be defined"
            print(f"  ✓ {name}")
        
        print("✓ All classes defined correctly")
        return True
    except Exception as e:
        print(f"✗ Class definition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hardware_profile_methods():
    """Test that HardwareProfile has the expected methods."""
    print("\nTesting HardwareProfile methods...")
    
    try:
        from kineticstack.core.kernel_synthesis import HardwareProfile
        
        # Test class methods exist
        assert hasattr(HardwareProfile, 'h100_profile'), "Should have h100_profile method"
        assert hasattr(HardwareProfile, 'a100_profile'), "Should have a100_profile method"
        
        # Test they're callable
        assert callable(HardwareProfile.h100_profile), "h100_profile should be callable"
        assert callable(HardwareProfile.a100_profile), "a100_profile should be callable"
        
        print("✓ HardwareProfile methods exist")
        return True
    except Exception as e:
        print(f"✗ HardwareProfile methods test failed: {e}")
        return False


def test_api_surface():
    """Test that the main API surface is as expected."""
    print("\nTesting API surface...")
    
    try:
        from kineticstack.core.autograd_sculpting import AutogradSculptor
        from kineticstack.core.kernel_synthesis import KernelSynthesizer
        from kineticstack.agents.sculptor import SculptorAgent
        from kineticstack.agents.kernel_synthesizer import KernelSynthesizerAgent
        from kineticstack.agents.invariant_guard import InvariantGuard
        from kineticstack.core.runtime import KineticRuntime
        
        # Test AutogradSculptor methods
        sculptor_methods = ['trace_model', 'analyze_memory_pressure', 'identify_checkpointing_candidates', 
                           'refactor_for_hbm3e', 'sculpt', 'get_memory_profile']
        for method in sculptor_methods:
            assert hasattr(AutogradSculptor, method), f"AutogradSculptor should have {method}"
        
        # Test KernelSynthesizer methods
        synthesizer_methods = ['estimate_register_pressure', 'generate_matmul_kernel', 
                              'generate_attention_kernel', 'synthesize', 'get_hardware_profile']
        for method in synthesizer_methods:
            assert hasattr(KernelSynthesizer, method), f"KernelSynthesizer should have {method}"
        
        # Test SculptorAgent methods
        sculptor_agent_methods = ['analyze_gradient_tape', 'rewrite_gradient_tape', 
                                  'apply_checkpointing', 'optimize_model', 'get_checkpoint_strategy', 
                                  'estimate_savings']
        for method in sculptor_agent_methods:
            assert hasattr(SculptorAgent, method), f"SculptorAgent should have {method}"
        
        # Test KernelSynthesizerAgent methods
        kernel_agent_methods = ['analyze_register_pressure', 'generate_llvm_ir', 
                               'compile_kernel', 'optimize_for_hardware', 'jit_compile', 
                               'get_telemetry', 'clear_cache']
        for method in kernel_agent_methods:
            assert hasattr(KernelSynthesizerAgent, method), f"KernelSynthesizerAgent should have {method}"
        
        # Test InvariantGuard methods
        guard_methods = ['register_rule', 'validate', 'validate_transformation', 
                        'validate_graph', 'get_violations', 'get_summary']
        for method in guard_methods:
            assert hasattr(InvariantGuard, method), f"InvariantGuard should have {method}"
        
        # Test KineticRuntime methods
        runtime_methods = ['optimize_model', 'get_telemetry', 'get_optimization_result', 
                          'clear_cache', 'print_status']
        for method in runtime_methods:
            assert hasattr(KineticRuntime, method), f"KineticRuntime should have {method}"
        
        print("✓ All API methods present")
        return True
    except Exception as e:
        print(f"✗ API surface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """Test that all expected files exist."""
    print("\nTesting file structure...")
    
    base_path = os.path.join(os.path.dirname(__file__), '..')
    
    expected_files = [
        'setup.py',
        'requirements.txt',
        'README.md',
        '.gitignore',
        'src/kineticstack/__init__.py',
        'src/kineticstack/core/__init__.py',
        'src/kineticstack/core/autograd_sculpting.py',
        'src/kineticstack/core/kernel_synthesis.py',
        'src/kineticstack/core/runtime.py',
        'src/kineticstack/agents/__init__.py',
        'src/kineticstack/agents/sculptor.py',
        'src/kineticstack/agents/kernel_synthesizer.py',
        'src/kineticstack/agents/invariant_guard.py',
        'examples/basic_usage.py',
    ]
    
    try:
        for file_path in expected_files:
            full_path = os.path.join(base_path, file_path)
            assert os.path.exists(full_path), f"File should exist: {file_path}"
            print(f"  ✓ {file_path}")
        
        print("✓ All expected files present")
        return True
    except Exception as e:
        print(f"✗ File structure test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("KineticStack Structure Validation")
    print("=" * 60)
    
    tests = [
        test_package_structure,
        test_module_imports,
        test_class_definitions,
        test_hardware_profile_methods,
        test_api_surface,
        test_file_structure,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
