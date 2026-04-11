"""
FLUX Metrics — collect and analyze runtime performance metrics.

Tracks per-instruction timing, register lifetimes, stack depth,
memory access patterns, and throughput.
"""
import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


@dataclass
class InstructionMetric:
    opcode: int
    mnemonic: str
    count: int
    total_cycles: int
    min_cycles: int
    max_cycles: int
    avg_cycles: float


@dataclass
class ExecutionMetrics:
    total_instructions: int
    total_cycles: int
    wall_time_us: float
    instructions_per_second: float
    register_writes: Dict[int, int]
    register_reads: Dict[int, int]
    max_stack_depth: int
    branch_taken: int
    branch_not_taken: int
    per_opcode: Dict[str, InstructionMetric]
    
    def to_markdown(self) -> str:
        lines = ["# FLUX Execution Metrics\n"]
        lines.append("## Summary")
        lines.append(f"- Instructions: {self.total_instructions:,}")
        lines.append(f"- Cycles: {self.total_cycles:,}")
        lines.append(f"- Wall time: {self.wall_time_us:.1f} μs")
        lines.append(f"- Throughput: {self.instructions_per_second:,.0f} IPS")
        lines.append(f"- Branch: {self.branch_taken} taken / {self.branch_not_taken} not taken")
        lines.append(f"- Max stack depth: {self.max_stack_depth}")
        
        lines.append("\n## Hot Opcodes")
        sorted_ops = sorted(self.per_opcode.values(), key=lambda x: -x.count)[:10]
        for m in sorted_ops:
            lines.append(f"- **{m.mnemonic}**: {m.count}x, avg {m.avg_cycles:.1f} cycles")
        
        lines.append("\n## Register Activity")
        for reg in sorted(set(list(self.register_writes.keys()) + list(self.register_reads.keys())))[:8]:
            w = self.register_writes.get(reg, 0)
            r = self.register_reads.get(reg, 0)
            lines.append(f"- R{reg}: {w} writes, {r} reads")
        
        return "\n".join(lines)


OP_NAMES = {
    0x00:"HALT",0x01:"NOP",0x08:"INC",0x09:"DEC",0x0B:"NEG",
    0x0C:"PUSH",0x0D:"POP",0x18:"MOVI",0x19:"ADDI",
    0x20:"ADD",0x21:"SUB",0x22:"MUL",0x23:"DIV",0x24:"MOD",
    0x2C:"CMP_EQ",0x2D:"CMP_LT",0x2E:"CMP_GT",
    0x3A:"MOV",0x3C:"JZ",0x3D:"JNZ",0x40:"MOVI16",
}

BRANCH_OPS = {0x3C, 0x3D}


class InstrumentedVM:
    """FLUX VM with instruction-level metrics collection."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._opcode_counts: Dict[int, int] = defaultdict(int)
        self._opcode_cycles: Dict[int, List[int]] = defaultdict(list)
        self._reg_writes: Dict[int, int] = defaultdict(int)
        self._reg_reads: Dict[int, int] = defaultdict(int)
        self._stack_depth = 0
        self._max_stack = 0
        self._branch_taken = 0
        self._branch_not_taken = 0
        self._total_instructions = 0
        self._total_cycles = 0
    
    def run(self, bytecode: List[int], initial_regs: Dict[int, int] = None) -> Tuple[Dict[int, int], ExecutionMetrics]:
        self.reset()
        regs = [0] * 64
        stack = [0] * 4096
        sp = 4096
        pc = 0
        halted = False
        
        if initial_regs:
            for k, v in initial_regs.items():
                regs[k] = v
        
        def sb(b): return b - 256 if b > 127 else b
        
        bc = bytes(bytecode)
        t0 = time.perf_counter_ns()
        
        while not halted and pc < len(bc) and self._total_instructions < 100000:
            op = bc[pc]
            self._total_instructions += 1
            self._opcode_counts[op] += 1
            cycle = 1
            
            if op == 0x00: halted = True; pc += 1
            elif op == 0x01: pc += 1
            elif op == 0x08: 
                rd = bc[pc+1]; self._reg_writes[rd] += 1; regs[rd] += 1; pc += 2
            elif op == 0x09:
                rd = bc[pc+1]; self._reg_writes[rd] += 1; regs[rd] -= 1; pc += 2
            elif op == 0x0B:
                rd = bc[pc+1]; self._reg_writes[rd] += 1; regs[rd] = -regs[rd]; pc += 2
            elif op == 0x0C:
                rd = bc[pc+1]; self._reg_reads[rd] += 1
                self._stack_depth += 1; self._max_stack = max(self._max_stack, self._stack_depth)
                sp -= 1; stack[sp] = regs[rd]; pc += 2
            elif op == 0x0D:
                rd = bc[pc+1]; self._reg_writes[rd] += 1
                self._stack_depth = max(0, self._stack_depth - 1)
                regs[rd] = stack[sp]; sp += 1; pc += 2
            elif op == 0x18:
                rd = bc[pc+1]; self._reg_writes[rd] += 1; regs[rd] = sb(bc[pc+2]); pc += 3; cycle = 2
            elif op == 0x19:
                rd = bc[pc+1]; self._reg_reads[rd] += 1; self._reg_writes[rd] += 1
                regs[rd] += sb(bc[pc+2]); pc += 3; cycle = 2
            elif op == 0x20:
                rd, rs1, rs2 = bc[pc+1], bc[pc+2], bc[pc+3]
                self._reg_reads[rs1] += 1; self._reg_reads[rs2] += 1; self._reg_writes[rd] += 1
                regs[rd] = regs[rs1] + regs[rs2]; pc += 4; cycle = 2
            elif op == 0x21:
                rd, rs1, rs2 = bc[pc+1], bc[pc+2], bc[pc+3]
                self._reg_reads[rs1] += 1; self._reg_reads[rs2] += 1; self._reg_writes[rd] += 1
                regs[rd] = regs[rs1] - regs[rs2]; pc += 4; cycle = 2
            elif op == 0x22:
                rd, rs1, rs2 = bc[pc+1], bc[pc+2], bc[pc+3]
                self._reg_reads[rs1] += 1; self._reg_reads[rs2] += 1; self._reg_writes[rd] += 1
                regs[rd] = regs[rs1] * regs[rs2]; pc += 4; cycle = 4
            elif op == 0x23:
                rd, rs1, rs2 = bc[pc+1], bc[pc+2], bc[pc+3]
                self._reg_reads[rs1] += 1; self._reg_reads[rs2] += 1; self._reg_writes[rd] += 1
                if regs[rs2] != 0: regs[rd] = regs[rs1] // regs[rs2]
                pc += 4; cycle = 8
            elif op == 0x24:
                rd, rs1, rs2 = bc[pc+1], bc[pc+2], bc[pc+3]
                self._reg_reads[rs1] += 1; self._reg_reads[rs2] += 1; self._reg_writes[rd] += 1
                if regs[rs2] != 0: regs[rd] = regs[rs1] % regs[rs2]
                pc += 4; cycle = 8
            elif op == 0x2C:
                rd, rs1, rs2 = bc[pc+1], bc[pc+2], bc[pc+3]
                self._reg_reads[rs1] += 1; self._reg_reads[rs2] += 1; self._reg_writes[rd] += 1
                regs[rd] = 1 if regs[rs1] == regs[rs2] else 0; pc += 4; cycle = 2
            elif op == 0x2D:
                rd, rs1, rs2 = bc[pc+1], bc[pc+2], bc[pc+3]
                self._reg_reads[rs1] += 1; self._reg_reads[rs2] += 1; self._reg_writes[rd] += 1
                regs[rd] = 1 if regs[rs1] < regs[rs2] else 0; pc += 4; cycle = 2
            elif op == 0x2E:
                rd, rs1, rs2 = bc[pc+1], bc[pc+2], bc[pc+3]
                self._reg_reads[rs1] += 1; self._reg_reads[rs2] += 1; self._reg_writes[rd] += 1
                regs[rd] = 1 if regs[rs1] > regs[rs2] else 0; pc += 4; cycle = 2
            elif op == 0x3A:
                rd, rs1 = bc[pc+1], bc[pc+2]; self._reg_reads[rs1] += 1; self._reg_writes[rd] += 1
                regs[rd] = regs[rs1]; pc += 4; cycle = 1
            elif op == 0x3C:
                rd = bc[pc+1]; self._reg_reads[rd] += 1
                if regs[rd] == 0: pc += sb(bc[pc+2]); self._branch_taken += 1
                else: pc += 4; self._branch_not_taken += 1
                cycle = 2
            elif op == 0x3D:
                rd = bc[pc+1]; self._reg_reads[rd] += 1
                if regs[rd] != 0: pc += sb(bc[pc+2]); self._branch_taken += 1
                else: pc += 4; self._branch_not_taken += 1
                cycle = 2
            else: pc += 1
            
            self._total_cycles += cycle
            self._opcode_cycles[op].append(cycle)
        
        wall_ns = time.perf_counter_ns() - t0
        wall_us = wall_ns / 1000.0
        ips = self._total_instructions / (wall_us / 1_000_000) if wall_us > 0 else 0
        
        per_opcode = {}
        for op, counts in self._opcode_cycles.items():
            name = OP_NAMES.get(op, f"0x{op:02x}")
            per_opcode[name] = InstructionMetric(
                opcode=op, mnemonic=name, count=len(counts),
                total_cycles=sum(counts), min_cycles=min(counts),
                max_cycles=max(counts), avg_cycles=sum(counts)/len(counts)
            )
        
        metrics = ExecutionMetrics(
            total_instructions=self._total_instructions,
            total_cycles=self._total_cycles,
            wall_time_us=wall_us, instructions_per_second=ips,
            register_writes=dict(self._reg_writes),
            register_reads=dict(self._reg_reads),
            max_stack_depth=self._max_stack,
            branch_taken=self._branch_taken,
            branch_not_taken=self._branch_not_taken,
            per_opcode=per_opcode,
        )
        
        return {i: regs[i] for i in range(16)}, metrics


# ── Tests ──────────────────────────────────────────────

import unittest


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.vm = InstrumentedVM()
    
    def test_basic_execution(self):
        regs, m = self.vm.run([0x18, 0, 42, 0x00])
        self.assertEqual(regs[0], 42)
        self.assertEqual(m.total_instructions, 2)
    
    def test_cycle_counting(self):
        _, m = self.vm.run([0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00])
        self.assertGreater(m.total_cycles, 0)
    
    def test_register_tracking(self):
        _, m = self.vm.run([0x18, 0, 10, 0x08, 0, 0x00])
        self.assertIn(0, m.register_writes)
        self.assertGreater(m.register_writes[0], 0)
    
    def test_branch_tracking(self):
        _, m = self.vm.run([0x18, 0, 0, 0x3C, 0, 1, 0, 0x18, 1, 1, 0x00])
        self.assertGreater(m.branch_taken + m.branch_not_taken, 0)
    
    def test_stack_depth(self):
        _, m = self.vm.run([0x0C, 0, 0x0C, 0, 0x0D, 0, 0x0D, 0, 0x00])
        self.assertGreater(m.max_stack_depth, 0)
    
    def test_markdown_report(self):
        _, m = self.vm.run([0x18, 0, 42, 0x00])
        md = m.to_markdown()
        self.assertIn("Summary", md)
        self.assertIn("IPS", md)
    
    def test_per_opcode_stats(self):
        _, m = self.vm.run([0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00])
        self.assertIn("MOVI", m.per_opcode)
        self.assertIn("ADD", m.per_opcode)
    
    def test_throughput(self):
        _, m = self.vm.run([0x18, 0, 42, 0x00])
        self.assertGreater(m.instructions_per_second, 0)
    
    def test_factorial(self):
        bc = [0x18,0,6, 0x18,1,1, 0x22,1,1,0, 0x09,0, 0x3D,0,0xFA,0, 0x00]
        regs, m = self.vm.run(bc)
        self.assertEqual(regs[1], 720)
        self.assertGreater(m.branch_taken, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
