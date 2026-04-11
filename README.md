# FLUX Metrics — Runtime Performance Analysis

Instrumented FLUX VM that collects instruction-level metrics.

## Features
- Per-opcode cycle counting (ADD=2, MUL=4, DIV=8 cycles)
- Register read/write tracking
- Stack depth monitoring
- Branch prediction stats (taken/not-taken)
- Wall-clock throughput (IPS)
- Markdown report generation

## Usage
```python
from metrics import InstrumentedVM
vm = InstrumentedVM()
regs, metrics = vm.run([0x18, 0, 42, 0x00])
print(metrics.to_markdown())
```

9 tests passing.
