"""
src/models/devs.py

Discrete Event System Specification (DEVS) Models and Simulator.


This module provides a pure-Python implementation of atomic models, 
coupled model routing, and the DEVS Coordinator algorithm.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

INFINITY = math.inf

# ==========================================
# 1. CORE DEVS DATA STRUCTURES
# ==========================================

class Message:
    """Represents an event passing between DEVS components."""
    def __init__(self, port: str, value: Any = None):
        self.port = port
        self.value = value


class DEVSAtomic:
    """
    Abstract base class for a DEVS Atomic Model (Section 3.5.1).
    Structure: <X, Y, S, delta_ext, delta_int, delta_con, lambda, ta>
    """
    def __init__(self, name: str):
        self.name = name
        self.phase = "passive"
        self.sigma = INFINITY
        
        # Internal tracking for the coordinator
        self.last_transition_time = 0.0
        self.next_event_time = INFINITY

    def hold_in(self, phase: str, sigma: float):
        """Helper to simultaneously update phase and sigma."""
        self.phase = phase
        self.sigma = sigma

    def initialize(self):
        """Initialization function defining the initial state."""
        raise NotImplementedError()

    def delta_int(self):
        """Internal transition function (delta_int). Triggered when sigma expires."""
        raise NotImplementedError()

    def delta_ext(self, e: float, msg: Message):
        """
        External transition function (delta_ext). 
        Triggered when an external message arrives.
        
        Args:
            e: Elapsed time since the last state transition.
            msg: The incoming Message object.
        """
        raise NotImplementedError()

    def delta_con(self, msg: Message):
        """
        Confluent transition function (delta_con).
        Triggered when internal and external events happen at the exact same time.
        Default implementation: Internal transition followed immediately by External.
        """
        self.delta_int()
        self.delta_ext(0.0, msg)

    def output_func(self) -> Optional[Message]:
        """Output function (lambda). Generates output just before delta_int."""
        return None

    def get_state(self) -> Dict[str, Any]:
        """Returns the full state vector (Required for Chapter 7 Data Assimilation)."""
        return {"phase": self.phase, "sigma": self.sigma}

    def set_state(self, state_dict: Dict[str, Any]):
        """Injects a full state vector (Required for Chapter 7 Particle Rejuvenation)."""
        self.phase = state_dict.get("phase", self.phase)
        self.sigma = state_dict.get("sigma", self.sigma)


# ==========================================
# 2. DEVS SIMULATION ALGORITHM (Coordinator)
# ==========================================

class DEVSCoordinator:
    """
    The DEVS Simulation Algorithm (Algorithm 3.3).
    Acts as the Simulation Controller described in Chapter 7 (Fig 7.5).
    """
    def __init__(self):
        self.models: List[DEVSAtomic] = []
        self.couplings: List[Tuple[str, str, str, str]] =[] # (src, src_port, dest, dest_port)
        self.current_time = 0.0

    def add_model(self, model: DEVSAtomic):
        self.models.append(model)

    def add_coupling(self, src_model: str, src_port: str, dest_model: str, dest_port: str):
        """Defines how outputs from one model route as inputs to another."""
        self.couplings.append((src_model, src_port, dest_model, dest_port))

    def initialize(self):
        """Step 1: Initialize all component models."""
        self.current_time = 0.0
        for m in self.models:
            m.initialize()
            m.last_transition_time = 0.0
            m.next_event_time = m.sigma

    def route_messages(self, outputs: Dict[str, Message]) -> Dict[str, Message]:
        """Routes outputs to destinations based on defined couplings."""
        inputs = {}
        for src_name, msg in outputs.items():
            if msg is None: continue
            for src, sport, dest, dport in self.couplings:
                if src == src_name and sport == msg.port:
                    # Create the mapped message for the destination
                    inputs[dest] = Message(port=dport, value=msg.value)
        return inputs

    def step(self):
        """Executes one single discrete event processing cycle."""
        # 1. Find the next event time (tN)
        tN = min(m.next_event_time for m in self.models)
        if tN == INFINITY:
            return False # Nothing left to do
            
        self.current_time = tN

        # 2. Identify Imminents (models scheduled to transition right now)
        imminents =[m for m in self.models if m.next_event_time == tN]

        # 3. Compute Outputs
        outputs = {m.name: m.output_func() for m in imminents}

        # 4. Route Messages (Determine Receivers)
        inputs = self.route_messages(outputs)

        # 5. Apply State Transitions
        for m in self.models:
            is_imminent = m in imminents
            has_input = m.name in inputs
            
            if is_imminent or has_input:
                e = self.current_time - m.last_transition_time
                
                if is_imminent and not has_input:
                    m.delta_int()
                elif not is_imminent and has_input:
                    m.delta_ext(e, inputs[m.name])
                elif is_imminent and has_input:
                    m.delta_con(inputs[m.name])
                
                # Update time tracking post-transition
                m.last_transition_time = self.current_time
                m.next_event_time = self.current_time + m.sigma

        return True

    def run_until(self, end_time: float):
        """
        Runs the simulation up to a specific time. 
        Crucial for the piecewise Simulation Controller in DDDS (Chapter 7).
        """
        while True:
            tN = min(m.next_event_time for m in self.models)
            if tN > end_time or tN == INFINITY:
                break
            self.step()
            
        self.current_time = end_time


# ==========================================
# 3.  EXAMPLE: CAR WASH SYSTEM
# ==========================================

class CarGenerator(DEVSAtomic):
    """Generates cars (Pseudocode 3.1)"""
    def __init__(self, name: str, inter_gen_time: float = 6.0):
        super().__init__(name)
        self.inter_gen_time = inter_gen_time

    def initialize(self):
        self.hold_in("active", self.inter_gen_time)

    def delta_ext(self, e: float, msg: Message):
        if msg.port == "stop":
            self.hold_in("passive", INFINITY)

    def delta_int(self):
        if self.phase == "active":
            self.hold_in("active", self.inter_gen_time)

    def output_func(self):
        if self.phase == "active":
            return Message(port="carOut", value="car")
        return None


class CarWashCenter(DEVSAtomic):
    """Processes cars (Pseudocode 3.2)"""
    def __init__(self, name: str, car_wash_time: float = 8.0):
        super().__init__(name)
        self.car_wash_time = car_wash_time

    def initialize(self):
        self.hold_in("idle", INFINITY)

    def delta_ext(self, e: float, msg: Message):
        if msg.port == "car" and self.phase == "idle":
            self.hold_in("busy", self.car_wash_time)
        elif msg.port == "stop":
            self.hold_in("passive", INFINITY)

    def delta_int(self):
        if self.phase == "busy":
            self.hold_in("idle", INFINITY)

    def output_func(self):
        if self.phase == "busy":
            return Message(port="finishedCarOut", value="car")
        return None


class Transducer(DEVSAtomic):
    """Monitors performance and halts simulation (Pseudocode 3.3)"""
    def __init__(self, name: str, monitor_time_window: float = 40.0):
        super().__init__(name)
        self.monitor_time_window = monitor_time_window
        self.num_arrived = 0
        self.num_finished = 0

    def initialize(self):
        self.hold_in("active", self.monitor_time_window)

    def delta_ext(self, e: float, msg: Message):
        if msg.port == "arrived":
            self.num_arrived += 1
        elif msg.port == "solved":
            self.num_finished += 1

    def delta_int(self):
        if self.phase == "active":
            print(f"\n[Transducer Results] Time: {self.current_time}")
            print(f"Cars Arrived: {self.num_arrived} | Cars Finished: {self.num_finished}")
            self.hold_in("passive", INFINITY)

    def output_func(self):
        if self.phase == "active":
            return Message(port="stopOut", value="stop")
        return None


# ==========================================
# 4. EXECUTABLE DEMONSTRATION 
# ==========================================

if __name__ == "__main__":
    print("Initializing DEVS Car Wash System (Section 3.5.3)...")
    
    # Instantiate Models
    gen = CarGenerator("Gen", inter_gen_time=6.0)
    wash = CarWashCenter("Wash", car_wash_time=8.0)
    monitor = Transducer("Monitor", monitor_time_window=40.0)

    # Configure Simulator
    sim = DEVSCoordinator()
    sim.add_model(gen)
    sim.add_model(wash)
    sim.add_model(monitor)

    # Map Couplings (Matches Figure 3.8)
    sim.add_coupling("Gen", "carOut", "Wash", "car")
    sim.add_coupling("Gen", "carOut", "Monitor", "arrived")
    sim.add_coupling("Wash", "finishedCarOut", "Monitor", "solved")
    sim.add_coupling("Monitor", "stopOut", "Gen", "stop")
    sim.add_coupling("Monitor", "stopOut", "Wash", "stop")

    # Run
    sim.initialize()
    sim.run_until(50.0) # Run slightly past the 40s monitor window to ensure stop logic executes
    print("Simulation Complete.")
