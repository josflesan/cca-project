from abc import ABC, abstractmethod
from typing import Dict, List
from collections import deque, defaultdict
from typing import Deque
from docker.models.containers import Container

from scheduler_logger import SchedulerLogger
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
    def on_job_complete(self, state: State) -> None:
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

    def on_state_update(self, state: State) -> None:
        """
        # transition High -> Low
            if we are on a 3 core job, scale up to 3 cores
            if there is a 3 core job, switch to it
            swith to 2 core job if there is one
            switch to 1 core if there is one
            unpause 1 core job

        """
        job_manager = state.job_queues
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
                    self.run(pending_2_cores, cores=state.cores_available)
                return
            elif running_3_cores:
                self.update(running_3_cores[0], cores=state.cores_available)
                return

            running_1_cores = job_manager.get_pending_by_core(1)
            running_2_cores = job_manager.get_pending_by_core(2)
            if running_2_cores and running_1_cores:
                self.pause(running_1_cores[0])
            if len(running_1_cores) == 3:  # unlikely
                self.pause(running_1_cores[0])

        # if running 3 core job, then pause it and run 2 core job
        # if running 3 core job, and only 1, reduce 3 core job to 2 cores
        # if running 1 and 2, pause the 1 core job
        else:
            # we never switch to a 2 core job here if im not mistaken, might be wrong its 2 am :((
            if benchmarks := job_manager.get_running_by_core(3):
                self.update(benchmarks[0], state.cores_available)
            elif benchmarks := job_manager.get_paused_by_core(3):
                for bc in job_manager.running:
                    self.pause(bc)
                # ! The core it runs on should be the responsibility of the strategy not the controller
                self.unpause(benchmarks[0])
            elif job_manager.get_running_by_core(
                2
            ) or not job_manager.get_paused_by_core(2):
                if benchmarks := job_manager.get_paused_by_core(1):
                    self.unpause(benchmarks[0])
                elif benchmarks := job_manager.get_pending_by_core(1):
                    self.run(benchmarks[0], cores=state.cores_available[0])
            else:
                raise RuntimeError("Should be unreachable")

    def on_job_complete(self, state: State):
        # can use inefficient operations here since this is only called 6 times
        job_manager = state.job_queues
        
        non_running = job_manager.get_non_running()
        num_cores = len(state.cores_available)
        bc = max(
            (bc for bc in non_running if bc.cores_num <= num_cores),
            key=lambda x: x.cores_num,
            default=max(non_running),
        )
        if bc.cores_num >= 2 :
            self.run(bc, cores=state.cores_available)
        else:
            # TODO: needs cleaning up lol
            if num_cores >= 2:
                if paused:= job_manager.get_running_by_core(3):
                    self.run(paused[0], cores=state.cores_available)
                elif pending := job_manager.get_running_by_core(3):
                    self.run(pending[0], cores=state.cores_available)
                else:
                    self.run(bc[0], cores=state.cores_available)
            else:
                self.run(bc[0], cores=state.cores_available)







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
