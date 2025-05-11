from abc import ABC, abstractmethod
from typing import Dict, List
from collections import deque, defaultdict
from typing import Deque
from docker.models.containers import Container

from scheduler_logger import SchedulerLogger
from utils.scheduler_utils import JobManager, State, Benchmark, Load, Pause, Unpause, Run, Update

class SchedulingStrategy(ABC):

    def __init__(self, ordering: Dict[str, int], colocations: Dict[str, List[str]] | None, logger: SchedulerLogger):
        self.ordering = ordering
        self.colocations = colocations
        self.logger = logger
        self.job_manager = JobManager([list, deque, deque])

        self.command_buffer = []  # Source of truth for commands ran on every state update

    @abstractmethod
    def on_state_update(self, state: State) -> None:
        pass

    @abstractmethod
    def on_job_complete(self, state: State) -> None:
        pass

    def pause(self, bench: Benchmark) -> None:
        self.command_buffer.append(Pause(bench))
    
    def unpause(self, bench: Benchmark) -> None:
        self.command_buffer.append(Unpause(bench))

    def run(self, bench: Benchmark, cores: List[int]) -> None:
        self.command_buffer.append(Run(bench, cores))

    def update(self, bench: Benchmark, cores: List[int]) -> None:
        self.command_buffer.append(Update(bench, cores))

# SCHEDULING STRATEGIES

class ShittyStrategy(SchedulingStrategy):
    """This is kinda shit but it works"""

    # I think that the job manager should be part of the strategy and not the state
    # Then we can control what those queues look like in terms of data structures
    # Would still be nice to have a method to call so we can pause/unpause and it abstract the complexity of those ops

    def __init__(self, ordering, colocations, logger):
        super().__init__(ordering, colocations, logger)
        
        # Convert pending list to pending dict
        self.pending: Dict[int, Deque[Benchmark]] = {
            1: deque([b for b in ordering if b.cores_num == 1]),
            2: deque([b for b in ordering if b.cores_num == 2]),
            3: deque([b for b in ordering if b.cores_num == 3]),
        }
        self.running: Dict[int, Deque[Container]] = defaultdict(deque)
        self.paused: Dict[int, Deque[Container]] = defaultdict(deque)

    def on_state_update(self, state: State):
        super().on_state_update(state)

        # If the load changes, pause running containers
        if state.load == Load.HIGH:
            # Pause all running containers
            for bench in state.running:
                self.pause(bench)
        else:
            # Pause 2 and 3 core containers
            for bench in state.running:
                self.pause(bench)

        # Get next job
        self.get_next_job(state)
        
    def get_next_job(self, state):
        job_manager = state.job_manager
        num_cores_available = state.get_num_cores_available()
        cores_available = state.get_cores_available()

        # Get relevant jobs
        paused_3_core = job_manager.get_paused_by_core(3)
        pending_3_core = job_manager.get_pending_by_core(3)
        paused_2_core = job_manager.get_paused_by_core(2)
        pending_2_core = job_manager.get_pending_by_core(2)
        paused_1_core = job_manager.get_paused_by_core(1)
        pending_1_core = job_manager.get_pending_by_core(1)

        if state.load == Load.LOW:
            

            # Check available cores and run highest in paused, fall back to pending otherwise
            if paused_3_core and num_cores_available == 3:
                bench = paused_3_core[0]
                self.unpause(bench)
            elif pending_3_core and num_cores_available == 3:
                bench = pending_3_core[0]
                self.run(bench, cores=cores_available)
            elif paused_2_core and num_cores_available >= 2:
                bench = paused_2_core[0]
                self.update(bench, cores_available[-2:])
                self.unpause(bench)
                




            
            elif self.paused[2] and num_cores_available >= 2:
                self.unpause_container(cores=2, cores_to_pin=cores_available[-2:])
            elif self.pending[2] and num_cores_available >= 2:
                self.schedule_next_benchmark(cores=2, cores_to_pin=cores_available[-2:])
            elif self.paused[1] and num_cores_available >= 1:
                self.unpause_container(cores=1, cores_to_pin=[cores_available[-1]])
            elif self.pending[1] and num_cores_available >= 1:
                self.schedule_next_benchmark(cores=1, cores_to_pin=[cores_available[-1]])

        else:
            
            # Check available cores and run highest in paused, fall back to pending
            if self.paused[2] and num_cores_available == 2:
                self.unpause_container(cores=2, cores_to_pin=cores_available)
            elif self.pending[2] and num_cores_available == 2:
                self.schedule_next_benchmark(cores=2, cores_to_pin=cores_available)
            elif self.paused[3] and num_cores_available == 2:
                self.unpause_container(cores=3, cores_to_pin=cores_available)
            elif self.pending[3] and num_cores_available == 2:
                self.schedule_next_benchmark(cores=3, cores_to_pin=cores_available)
            elif self.paused[1] and num_cores_available >= 1:
                self.unpause_container(cores=1, cores_to_pin=[cores_available[-1]])
            elif self.pending[1] and num_cores_available >= 1:
                self.schedule_next_benchmark(cores=1, cores_to_pin=[cores_available[-1]])
    
    def on_job_complete(self, state: State):
        return super().on_job_complete(state)

class LongestJobFirst(SchedulingStrategy):
    """No interference information, just runs jobs sequentially from longest to shortest"""
    
    def on_state_update(self, state):
        return super().on_state_update(state)
    
    def on_job_complete(self, state):
        return super().on_job_complete(state)


class ShortestJobFirst(SchedulingStrategy):
    """No interference information, just runs jobs sequentially from shortest to longest"""
    
    def on_state_update(self, state):
        return super().on_state_update(state)
    
    def on_job_complete(self, state):
        return super().on_job_complete(state)


class InterferenceAwarePause(SchedulingStrategy):
    """Uses interference information and pauses running containers"""

    def on_state_update(self, state):
        return super().on_state_update(state)
    
    def on_job_complete(self, state):
        return super().on_job_complete(state)

class InterferenceAwareScaling(SchedulingStrategy):
    """Uses interference information but does not pause containers, instead scaling core requirements"""
    
    def on_state_update(self, state):
        return super().on_state_update(state)
    
    def on_job_complete(self, state):
        return super().on_job_complete(state)
