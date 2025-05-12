from abc import ABC, abstractmethod
from typing import Dict, List
from collections import deque, defaultdict
from typing import Deque
from docker.models.containers import Container

from utils.scheduler_logger import SchedulerLogger
from utils.scheduler_utils import (
    JobManager,
    State,
    Benchmark,
    Load,
    Pause,
    Unpause,
    Run,
    Update,
)


class SchedulingStrategy(ABC):

    def __init__(
        self,
        ordering: Dict[str, int],
        colocations: Dict[str, List[str]] | None,
        logger: SchedulerLogger,
    ):
        self.ordering = ordering
        self.colocations = colocations
        self.logger = logger
        self.command_buffer = (
            []
        )  # Source of truth for commands ran on every state update

    @abstractmethod
    def on_state_update(self, state: State) -> None:
        pass

    @abstractmethod
    def on_job_complete(self, completed_jobs: List[Benchmark], state: State) -> None:
        pass

    def pause(self, benchmark: Benchmark) -> None:
        self.command_buffer.append(Pause(benchmark))

    def unpause(self, benchmark: Benchmark) -> None:
        self.command_buffer.append(Unpause(benchmark))

    def run(self, benchmark: Benchmark, cores: List[int]) -> None:
        self.command_buffer.append(Run(benchmark, cores))

    def update(self, benchmark: Benchmark, cores: List[int]) -> None:
        self.command_buffer.append(Update(benchmark, cores))


# SCHEDULING STRATEGIES


class ShittyStrategy(SchedulingStrategy):
    """This is kinda shit but it works"""

    # I think that the job manager should be part of the strategy and not the state
    # Then we can control what those queues look like in terms of data structures
    # Would still be nice to have a method to call so we can pause/unpause and it abstract the complexity of those ops

    def __init__(self, ordering=None, colocations=None, logger=None):
        super().__init__(ordering, colocations, logger)

    def _schedule_optimal_job(self, state: State, cores_available) -> Benchmark | None:
        """
        Schedule the Optimal Job according to Cores Available:

        - If we have 3 cores available
            If 3 core jobs remain in paused/pending, do those
            If only 2 core jobs remain in paused/pending, do those
            If only 1 core jobs remain in paused/pending, do those

        - If we have 2 cores available
            If 3 core jobs remain in paused/pending, downscale and run
            If only 2 core jobs remain in paused/pending, do those
            If only 1 core jobs remain in paused/pending, do those

        - If we have 1 core available
            If only 2 core jobs remain in paused/pending, do those
        """
        job_manager = state.job_manager
        core1_remaining = (job_manager.get_paused_by_core(1) + job_manager.get_pending_by_core(1))
        core2_remaining = (job_manager.get_paused_by_core(2) + job_manager.get_pending_by_core(2))
        core3_remaining = (job_manager.get_paused_by_core(3) + job_manager.get_pending_by_core(3))
        # this could be a max but whatever
        if cores_available == 3:
            if core3_remaining:
                return core3_remaining[0]
            elif core2_remaining:
                return core2_remaining[0]
            elif core1_remaining:
                return core1_remaining[0]
        
        # We should favour 2 cores instead of 3 cores in this case no ?
        # example, if we are on high load and a 2 completes, surely we want the 2 core job to run instead of downscaling a 3 core job
        elif cores_available == 2:
            if core2_remaining:
                return core2_remaining[0]
            elif core1_remaining:
                return core1_remaining[0]
            elif core3_remaining:
                self.update(core3_remaining[0], cores=state.cores_available)
                return core3_remaining[0]
        
        elif cores_available == 1:
            # same here, are we sure about favouring running a 2 core job over a 1 core job if there is core available
            # Here we don't want to try run a 3-core job on 1 single core
            if core1_remaining:
                return core1_remaining[0]
            elif core2_remaining:
                self.update(core2_remaining[0], cores=state.cores_available)
                return core2_remaining[0]
            
        return None

    def on_state_update(self, state: State) -> None:
        job_manager = state.job_manager

        # if running 3 core job, then pause it and run 2 core job
        # if running 3 core job, and only 1, reduce 3 core job to 2 cores
        # if running 1 and 2, pause the 1 core job
        if state.load == Load.HIGH:
            # transition Low -> High
            running_3_cores = job_manager.get_running_by_core(3)
            paused_2_cores = job_manager.get_paused_by_core(2)
            pending_2_cores = job_manager.get_pending_by_core(2)
            if running_3_cores and (paused_2_cores or pending_2_cores):
                self.pause(running_3_cores[0])
                if paused_2_cores:
                    self.unpause(paused_2_cores[0])
                else:
                    self.run(pending_2_cores[0], cores=state.cores_available)
                return
            elif running_3_cores:
                self.update(running_3_cores[0], cores=state.cores_available)
                return

            running_1_cores = job_manager.get_pending_by_core(1)
            running_2_cores = job_manager.get_pending_by_core(2)
            if running_2_cores and running_1_cores:
                self.pause(running_1_cores[0])
            elif len(running_1_cores) == 3:  # unlikely
                self.pause(running_1_cores[0])

        # if we are on a 3 core job, scale up to 3 cores
        # if there is a 3 core job, switch to it
        # if 2 core job running, co-schedule a 1-core
            # if no 1-core available, co-schedule a 2-core and downscale
        # if 1 core job running and 1 more pending, spin up an extra 1-core job 
        else:
            # we never switch to a 2 core job here if im not mistaken, might be wrong its 2 am :((
            if benchmarks := job_manager.get_running_by_core(3):
                self.update(benchmarks[0], state.cores_available)
            elif benchmarks := job_manager.get_paused_by_core(3):
                for bc in job_manager.running:
                    self.pause(bc)
                # ! The core it runs on should be the responsibility of the strategy not the controller
                self.unpause(benchmarks[0])
            elif job_manager.get_running_by_core(2):
                if benchmarks := job_manager.get_paused_by_core(1):
                    self.unpause(benchmarks[0])
                elif benchmarks := job_manager.get_pending_by_core(1):
                    self.run(benchmarks[0], cores=state.cores_available[0])
                elif benchmarks := job_manager.get_paused_by_core(2):
                    self.update(benchmarks[0], state.cores_available)
                    self.unpause(benchmarks[0])
                elif benchmarks := job_manager.get_pending_by_core(2):
                    self.run(benchmarks[0], state.cores_available)
            elif job_manager.get_running_by_core(1):
                if benchmarks := job_manager.get_paused_by_core(1):
                    self.unpause(benchmarks[0])
                elif benchmarks := job_manager.get_pending_by_core(1):
                    self.run(benchmarks[0], state.cores_available[-1])
            else:
                raise RuntimeError("Should be unreachable")
    
    def on_job_complete(self, copleted_jobs, state: State):        
        # can use inefficient operations here since this is only called 6 times
        num_cores_available = len(state.cores_available)
        
        while bench := self._schedule_optimal_job(state, num_cores_available):
            # Run or unpause the benchmark
            if bench.is_paused():
                self.unpause(bench)
            else:
                self.run(bench, cores=state.cores_available[-bench.cores_num:])
            
            # Update remaining cores
            num_cores_available -= bench.cores_num


class NoPauseStrategy(SchedulingStrategy):
    """No interference information, run without pausing containers
    
    - Transition low -> high
        if running 3 core job downscale to 2 core
        if running 2 core and 1 core, colocate
        if running 3 1-core jobs, colocate
        # if running two 2 core jobs, colocate // if things are ordered well this shouldnt happen
    
    - Transition high -> low:
        if running 3 core, upscale to 3 ore
        if running 2 and 1, move 1 core to free core
        if only running 2 check if theres pending 1 and running    
        if running colocated 2 core jobs, make them overlap only in 1 core
    """

    def on_state_update(self, state: State) -> None:
        job_manager = state.job_manager
        if state.load == Load.HIGH:
            running_3c = job_manager.get_running_by_core(3)
            if running_3c:
                self.update(running_3c[0], cores=state.cores_available)
                return
            
            running_2c = job_manager.get_running_by_core(2)
            running_1c = job_manager.get_running_by_core(1)
            if running_1c and running_2c:
                self.update(running_1c[0], cores=[2])
                return
            
            if len(running_1c) == 3:
                for benchmark in running_1c:
                    cores = benchmark.container.attrs["HostConfig"]["CpusetCpus"].split("-")
                    if "1" in cores:
                        self.update(benchmark, cores=[2])
                return


            if len(running_2c) == 2:
                self.update(running_2c[0], cores=[2, 3])
                self.update(running_2c[1], cores=[2, 3])


        else:
            
            running_3c = job_manager.get_running_by_core(3)
            if running_3c:
                self.update(running_3c[0], cores=state.cores_available)

            running_2c = job_manager.get_running_by_core(2)
            running_1c = job_manager.get_running_by_core(1)
            pending_1c = job_manager.get_pending_by_core(1)
            if running_2c:

                if running_1c:
                    self.update(running_1c[0], cores=[1])
                if pending_1c:
                    self.run(pending_1c[0], cores=[1])

            pass

    def on_job_complete(self, completed_jobs, state: State) -> None:
        job_manager = state.job_manager
        if state.cores_available:
            next_benchmark = job_manager.get_highest_pending()
            cores_to_occupy = min(len(state.cores_available), next_benchmark.cores_num)
            self.run(next_benchmark, state.cores_available[-cores_to_occupy:])
        # just schedule next highest prio job


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
