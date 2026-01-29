"""
Invariant Guard: Validation and safety checks for the self-evolving stack.
Ensures that optimizations maintain correctness and numerical stability.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.fx as fx


class ViolationSeverity(Enum):
    """Severity levels for invariant violations"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class InvariantViolation:
    """Record of an invariant violation"""
    rule_name: str
    severity: ViolationSeverity
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class InvariantRule:
    """Definition of an invariant rule"""
    name: str
    description: str
    check_function: Callable
    severity: ViolationSeverity = ViolationSeverity.WARNING


class InvariantGuard:
    """
    Invariant Guard: Validation and Safety System
    
    Monitors and validates that optimizations and transformations maintain
    correctness, numerical stability, and expected behavior.
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the Invariant Guard.
        
        Args:
            strict_mode: If True, treat warnings as errors
        """
        self.strict_mode = strict_mode
        self.violations: List[InvariantViolation] = []
        self.rules: Dict[str, InvariantRule] = {}
        self._register_default_rules()
        
    def _register_default_rules(self):
        """Register default invariant rules."""
        
        # Rule: Output shape consistency
        self.register_rule(
            name="output_shape_consistency",
            description="Output shapes must match between original and optimized models",
            check_function=self._check_output_shape,
            severity=ViolationSeverity.ERROR,
        )
        
        # Rule: Numerical stability
        self.register_rule(
            name="numerical_stability",
            description="Outputs must be numerically stable (no NaN or Inf)",
            check_function=self._check_numerical_stability,
            severity=ViolationSeverity.CRITICAL,
        )
        
        # Rule: Memory bounds
        self.register_rule(
            name="memory_bounds",
            description="Memory usage must stay within allocated bounds",
            check_function=self._check_memory_bounds,
            severity=ViolationSeverity.ERROR,
        )
        
        # Rule: Gradient flow
        self.register_rule(
            name="gradient_flow",
            description="Gradients must flow properly through checkpointed operations",
            check_function=self._check_gradient_flow,
            severity=ViolationSeverity.WARNING,
        )
        
        # Rule: Type consistency
        self.register_rule(
            name="type_consistency",
            description="Data types must be preserved through transformations",
            check_function=self._check_type_consistency,
            severity=ViolationSeverity.WARNING,
        )
    
    def register_rule(
        self,
        name: str,
        description: str,
        check_function: Callable,
        severity: ViolationSeverity = ViolationSeverity.WARNING,
    ):
        """
        Register a new invariant rule.
        
        Args:
            name: Unique name for the rule
            description: Human-readable description
            check_function: Function that performs the check
            severity: Severity level for violations
        """
        self.rules[name] = InvariantRule(
            name=name,
            description=description,
            check_function=check_function,
            severity=severity,
        )
    
    def _check_output_shape(self, context: Dict[str, Any]) -> Optional[str]:
        """Check if output shapes are consistent."""
        original_shape = context.get('original_shape')
        optimized_shape = context.get('optimized_shape')
        
        if original_shape is None or optimized_shape is None:
            return None
        
        if original_shape != optimized_shape:
            return f"Shape mismatch: original {original_shape} != optimized {optimized_shape}"
        
        return None
    
    def _check_numerical_stability(self, context: Dict[str, Any]) -> Optional[str]:
        """Check for numerical stability issues."""
        output = context.get('output')
        
        if output is None:
            return None
        
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                return "Output contains NaN values"
            if torch.isinf(output).any():
                return "Output contains Inf values"
        
        return None
    
    def _check_memory_bounds(self, context: Dict[str, Any]) -> Optional[str]:
        """Check if memory usage is within bounds."""
        memory_used = context.get('memory_used_bytes', 0)
        memory_budget = context.get('memory_budget_bytes') or float('inf')
        
        if memory_used > memory_budget:
            return f"Memory usage {memory_used / 1e9:.2f}GB exceeds budget {memory_budget / 1e9:.2f}GB"
        
        return None
    
    def _check_gradient_flow(self, context: Dict[str, Any]) -> Optional[str]:
        """Check if gradients flow properly."""
        has_grad = context.get('has_gradient', True)
        requires_grad = context.get('requires_grad', False)
        
        if requires_grad and not has_grad:
            return "Gradient flow interrupted: requires_grad=True but no gradient computed"
        
        return None
    
    def _check_type_consistency(self, context: Dict[str, Any]) -> Optional[str]:
        """Check if data types are consistent."""
        original_dtype = context.get('original_dtype')
        optimized_dtype = context.get('optimized_dtype')
        
        if original_dtype is None or optimized_dtype is None:
            return None
        
        if original_dtype != optimized_dtype:
            return f"Type mismatch: original {original_dtype} != optimized {optimized_dtype}"
        
        return None
    
    def validate(self, context: Dict[str, Any], rules: Optional[List[str]] = None) -> List[InvariantViolation]:
        """
        Validate context against invariant rules.
        
        Args:
            context: Dictionary containing validation context
            rules: List of rule names to check (None = check all)
            
        Returns:
            List of violations found
        """
        violations = []
        
        rules_to_check = rules if rules else list(self.rules.keys())
        
        for rule_name in rules_to_check:
            if rule_name not in self.rules:
                continue
            
            rule = self.rules[rule_name]
            
            try:
                error_msg = rule.check_function(context)
                
                if error_msg:
                    violation = InvariantViolation(
                        rule_name=rule_name,
                        severity=rule.severity,
                        message=error_msg,
                        context=context.copy(),
                    )
                    violations.append(violation)
                    self.violations.append(violation)
            except Exception as e:
                # If check function fails, record it as a violation
                violation = InvariantViolation(
                    rule_name=rule_name,
                    severity=ViolationSeverity.ERROR,
                    message=f"Check function failed: {str(e)}",
                    context=context.copy(),
                )
                violations.append(violation)
                self.violations.append(violation)
        
        return violations
    
    def validate_transformation(
        self,
        original_output: torch.Tensor,
        optimized_output: torch.Tensor,
        memory_used: Optional[int] = None,
        memory_budget: Optional[int] = None,
    ) -> bool:
        """
        Validate a transformation between original and optimized models.
        
        Args:
            original_output: Output from original model
            optimized_output: Output from optimized model
            memory_used: Memory used by optimization (bytes)
            memory_budget: Memory budget (bytes)
            
        Returns:
            True if validation passes, False otherwise
        """
        context = {
            'original_shape': original_output.shape,
            'optimized_shape': optimized_output.shape,
            'original_dtype': original_output.dtype,
            'optimized_dtype': optimized_output.dtype,
            'output': optimized_output,
        }
        
        if memory_used is not None:
            context['memory_used_bytes'] = memory_used
        if memory_budget is not None:
            context['memory_budget_bytes'] = memory_budget
        
        violations = self.validate(context)
        
        # Check if any critical or error violations occurred
        has_critical = any(v.severity == ViolationSeverity.CRITICAL for v in violations)
        has_error = any(v.severity == ViolationSeverity.ERROR for v in violations)
        has_warning = any(v.severity == ViolationSeverity.WARNING for v in violations)
        
        if has_critical or has_error:
            return False
        
        if self.strict_mode and has_warning:
            return False
        
        return True
    
    def validate_graph(self, graph: fx.Graph) -> bool:
        """
        Validate a torch.fx graph for correctness.
        
        Args:
            graph: Graph to validate
            
        Returns:
            True if graph is valid, False otherwise
        """
        violations = []
        
        # Check for cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for user in node.users:
                if user not in visited:
                    if has_cycle(user):
                        return True
                elif user in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph.nodes:
            if node not in visited:
                if has_cycle(node):
                    violation = InvariantViolation(
                        rule_name="graph_acyclic",
                        severity=ViolationSeverity.CRITICAL,
                        message="Graph contains cycles",
                        context={'node': node.name},
                    )
                    violations.append(violation)
                    self.violations.append(violation)
        
        return len(violations) == 0
    
    def get_violations(
        self,
        severity: Optional[ViolationSeverity] = None,
    ) -> List[InvariantViolation]:
        """
        Get recorded violations, optionally filtered by severity.
        
        Args:
            severity: Filter by severity level (None = all)
            
        Returns:
            List of violations
        """
        if severity is None:
            return self.violations
        
        return [v for v in self.violations if v.severity == severity]
    
    def clear_violations(self):
        """Clear all recorded violations."""
        self.violations.clear()
    
    def get_summary(self) -> Dict[str, int]:
        """
        Get a summary of violations by severity.
        
        Returns:
            Dictionary mapping severity levels to counts
        """
        summary = {
            'critical': 0,
            'error': 0,
            'warning': 0,
            'info': 0,
            'total': len(self.violations),
        }
        
        for violation in self.violations:
            summary[violation.severity.value] += 1
        
        return summary
