from abc import ABC, abstractmethod
from typing import Dict, List

from utils.scheduler_logger import SchedulerLogger, Job
from utils.scheduler_utils import (
    JobManager,
    State,
    Benchmark,
    Load,
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
        
    @abstractmethod
    def on_state_update(self, state: State, job_manager: JobManager) -> None:
        pass

    @abstractmethod
    def on_job_complete(self, completed_jobs: List[Benchmark], state: State, job_manager: JobManager) -> None:
        pass

    def pause(self, benchmark: Benchmark, state: State, job_manager: JobManager) -> None:      
        did_pause = job_manager.pause(benchmark, state)

        if did_pause:
            self.logger.job_pause(Job._member_map_[benchmark.name.upper()])

    def unpause(self, benchmark: Benchmark, state: State, job_manager: JobManager) -> None:
        did_unpause = job_manager.unpause(benchmark, state)

        if did_unpause:
            self.logger.job_unpause(Job._member_map_[benchmark.name.upper()])

    def run(self, benchmark: Benchmark, cores: List[int], state: State, job_manager: JobManager) -> None:
        # print(f"CORES REQUESTED: {cores}")
        job_manager.run(benchmark, cores, state)
        self.logger.job_start(benchmark.to_job(), initial_cores=cores, initial_threads=benchmark.thread_num)
        
    def update(self, benchmark: Benchmark, cores: List[int], state: State, job_manager: JobManager) -> None:
        job_manager.update(benchmark, cores, state)
        self.logger.update_cores(Job._member_map_[benchmark.name.upper()], cores=cores)


# SCHEDULING STRATEGIES


class IsolationStrategy(SchedulingStrategy):
    """Try to run all jobs isolated, pausing and unpausing if impossible"""

    # I think that the job manager should be part of the strategy and not the state
    # Then we can control what those queues look like in terms of data structures
    # Would still be nice to have a method to call so we can pause/unpause and it abstract the complexity of those ops

    def __init__(self, ordering=None, colocations=None, logger=None):
        super().__init__(ordering, colocations, logger)

    def _schedule_optimal_job(self, state: State, job_manager: JobManager, num_cores_available: int) -> Benchmark | None:
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
        core1_remaining = (job_manager.get_paused_by_core(1) + job_manager.get_pending_by_core(1))
        core2_remaining = (job_manager.get_paused_by_core(2) + job_manager.get_pending_by_core(2))
        core3_remaining = (job_manager.get_paused_by_core(3) + job_manager.get_pending_by_core(3))
        # this could be a max but whatever
        if num_cores_available == 3:
            if core3_remaining:
                return core3_remaining[0]
            elif core2_remaining:
                return core2_remaining[0]
            elif core1_remaining:
                return core1_remaining[0]
        
        # We should favour 2 cores instead of 3 cores in this case no ?
        # example, if we are on high load and a 2 completes, surely we want the 2 core job to run instead of downscaling a 3 core job
        elif num_cores_available == 2:
            if core2_remaining:
                return core2_remaining[0]
            elif core3_remaining:
                if core3_remaining[0] in job_manager.paused:
                    self.update(core3_remaining[0], state.cores_available, state, job_manager)

                return core3_remaining[0]
            elif core1_remaining:
                return core1_remaining[0]
            
        elif num_cores_available == 1:
            # same here, are we sure about favouring running a 2 core job over a 1 core job if there is core available
            # Here we don't want to try run a 3-core job on 1 single core
            if core1_remaining:
                return core1_remaining[0]
            elif core2_remaining:
                if core2_remaining[0] in job_manager.paused:
                    self.update(core2_remaining[0], state.cores_available, state, job_manager)
                return core2_remaining[0]
            
        return None

    def on_state_update(self, state: State, job_manager: JobManager) -> None:

        # if running 3 core job, then pause it and run 2 core job
        # if running 3 core job, and only 1, reduce 3 core job to 2 cores
        # if running 1 and 2, pause the 1 core job
        if state.load == Load.HIGH:
            # transition Low -> High
            running_3_cores = job_manager.get_running_by_core(3)
            paused_2_cores = job_manager.get_paused_by_core(2)
            pending_2_cores = job_manager.get_pending_by_core(2)
            if running_3_cores and (paused_2_cores or pending_2_cores):
                self.pause(running_3_cores[0], state, job_manager)
                if paused_2_cores:
                    self.unpause(paused_2_cores[0], state, job_manager)
                else:
                    self.run(pending_2_cores[0], [2, 3], state, job_manager)
                return
            elif running_3_cores:
                # we need to relinquish cores for 3 cores
                self.update(running_3_cores[0], [2, 3], state, job_manager)
                return

            running_1_cores = job_manager.get_running_by_core(1)
            running_2_cores = job_manager.get_running_by_core(2)
            if running_2_cores and running_1_cores:
                self.pause(running_1_cores[0], state, job_manager)
            
            elif len(running_1_cores) == 3:  # unlikely
                # Pause the job that is running on core 1
                to_pause = job_manager.get_jobs_on_core("1")

                for bench in to_pause:
                    self.pause(running_1_cores[0], state, job_manager)

        # if we are on a 3 core job, scale up to 3 cores
        # if there is a 3 core job, switch to it
        # if 2 core job running, co-schedule a 1-core
            # if no 1-core available, co-schedule a 2-core and downscale
        # if 1 core job running and 1 more pending, spin up an extra 1-core job 
        else:

            # we never switch to a 2 core job here if im not mistaken, might be wrong its 2 am :((
            if benchmarks := job_manager.get_running_by_core(3):
                self.update(benchmarks[0], [1, 2, 3], state, job_manager)
            elif benchmarks := job_manager.get_paused_by_core(3):
                for bc in job_manager.running:
                    self.pause(bc, state, job_manager)
                
                self.unpause(benchmarks[0], state, job_manager)
            elif benchmarks := job_manager.get_pending_by_core(3):
                for bc in job_manager.running:
                    self.pause(bc, state, job_manager)
                self.run(benchmarks[0], state.cores_available, state, job_manager)
            
            elif job_manager.get_running_by_core(2):
                if benchmarks := job_manager.get_paused_by_core(1):
                    self.unpause(benchmarks[0], state, job_manager)
                elif benchmarks := job_manager.get_pending_by_core(1):
                    self.run(benchmarks[0], [state.cores_available[0]], state, job_manager)
                elif benchmarks := job_manager.get_paused_by_core(2):
                    # Downscale if there is a second 2-core job remaining and we are on LOW load
                    self.update(benchmarks[0], [state.cores_available[0]], state, job_manager)
                    self.unpause(benchmarks[0], state, job_manager)
                elif benchmarks := job_manager.get_pending_by_core(2):
                    self.run(benchmarks[0], [state.cores_available[0]], state, job_manager)

            elif job_manager.get_running_by_core(1):
                if benchmarks := job_manager.get_paused_by_core(1):
                    self.unpause(benchmarks[0], state, job_manager)
                elif benchmarks := job_manager.get_pending_by_core(1):
                    self.run(benchmarks[0], [state.cores_available[-1]], state, job_manager)
            else:
                raise RuntimeError("Should be unreachable")
    
    def on_job_complete(self, completed_jobs: List[Benchmark], state: State, job_manager: JobManager):        
        # can use inefficient operations here since this is only called 6 times
        num_cores_available = len(state.cores_available)
        
        while bench := self._schedule_optimal_job(state, job_manager, num_cores_available):
            # Run or unpause the benchmark
            if bench in job_manager.paused:
                self.unpause(bench, state, job_manager)
            else:
                cores_to_occupy = min(num_cores_available, bench.cores_num)
                self.run(bench, cores=state.cores_available[-cores_to_occupy:], state=state, job_manager=job_manager)
            
            # Update remaining cores
            num_cores_available -= bench.cores_num


class ScalingStrategy(SchedulingStrategy):
    """No interference information, run without pausing containers by scaling/downscaling containers
    
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

    def __init__(self, ordering=None, colocations=None, logger=None):
        super().__init__(ordering, colocations, logger)

    def _schedule_optimal_job(self, job_manager: JobManager, num_cores_available: int) -> Benchmark | None:
        """
        Schedule the next job in the pending list that satisfies the current state
        """
        next_bench_idx = 0
        core1_remaining = job_manager.get_pending_by_core(1)
        core2_remaining = job_manager.get_pending_by_core(2)
        core3_remaining = job_manager.get_pending_by_core(3)

        if num_cores_available == 3:        
            # Find the next benchmark in the ordering (pending list) that fits our current state
            # This is the optimal benchmark to run
            while next_bench_idx < len(job_manager.pending) and job_manager.pending[next_bench_idx].cores_num > num_cores_available:
                next_bench_idx += 1

            if next_bench_idx < len(job_manager.pending):
                return job_manager.pending[next_bench_idx]

        elif num_cores_available == 2:
            while next_bench_idx < len(job_manager.pending) and job_manager.pending[next_bench_idx].cores_num > num_cores_available:
                next_bench_idx += 1

            if next_bench_idx < len(job_manager.pending):
                return job_manager.pending[next_bench_idx]
            elif next_bench_idx >= len(job_manager.pending) and core3_remaining: 
                # Downscale a 3 if nothing below remains     
                return core3_remaining[0]

        elif num_cores_available == 1:
            if core1_remaining:
                return core1_remaining[0]
            elif core2_remaining:
                # Downscale a 2 if nothing below remains
                return core2_remaining[0]
            
            # We dont want to schedule a 3 core job on 1 core i think?
        
        return None

    def on_state_update(self, state: State, job_manager: JobManager) -> None:
        if state.load == Load.HIGH:
            running_3c = job_manager.get_running_by_core(3)
            if running_3c:
                # Downscale the 3 core job to 2 cores
                self.update(running_3c[0], cores=[2, 3], state=state, job_manager=job_manager)
                return
            
            running_2c = job_manager.get_running_by_core(2)
            running_1c = job_manager.get_running_by_core(1)
            if running_1c and running_2c:
                # If running a 2 and a 2, collocate the 1 core on the same cores as 2 core
                self.update(running_1c[0], cores=[2], state=state, job_manager=job_manager)
                return
            
            if len(running_1c) == 3:
                # If 3 1 core jobs were running...
                jobs_to_move = job_manager.get_jobs_on_core("1")
                self.update(jobs_to_move[0], cores=[2], state=state, job_manager=job_manager)
                
                return


            if len(running_2c) == 2:
                # If two 2 core jobs were running (downscaled) we downscale further to a single core each 
                self.update(running_2c[0], cores=[2], state=state, job_manager=job_manager)
                self.update(running_2c[1], cores=[3], state=state, job_manager=job_manager)

        else:
            running_3c = job_manager.get_running_by_core(3)
            if running_3c:
                # If running downscaled 3 core, upscale 
                self.update(running_3c[0], cores=[1, 2, 3], state=state, job_manager=job_manager)
                return

            running_2c = job_manager.get_running_by_core(2)
            running_3c = job_manager.get_running_by_core(3)
            running_1c = job_manager.get_running_by_core(1)
            pending_1c = job_manager.get_pending_by_core(1)
            pending_2c = job_manager.get_pending_by_core(2)
            pending_3c = job_manager.get_pending_by_core(3)
            if running_2c:
                # If the 2 core job had been downscaled (because 2 2-core running), we upscale again
                if len(running_2c) == 2:
                    self.update(running_2c[0], cores=[1, 2], state=state, job_manager=job_manager)
                elif running_1c:
                    # And running 1 core job, remove co-scheduling
                    self.update(running_1c[0], cores=[1], state=state, job_manager=job_manager)
                elif pending_1c:
                    # Or if 1 core jobs need running, kick them off
                    self.run(pending_1c[0], cores=[1], state=state, job_manager=job_manager)
                return
            
            if running_1c:

                if len(running_1c) == 3:
                    #TODO: maybe we can check the cores_available mapping instead
                    # If 3 1 core jobs are running, we now de-collocate the one running on core 2s
                    jobs_to_move = job_manager.get_jobs_on_core("2")

                    for bench in jobs_to_move:
                        # Move the one that was running on memcached core to safe core
                        self.update(bench, cores=[1], state=state, job_manager=job_manager)
                        break
                
                # If there is a pending 2, we run it on 2 cores
                elif len(running_1c) == 1 and pending_2c:
                    self.update(running_1c[0], cores=[1], state=state, job_manager=job_manager)
                    self.run(pending_2c[0], cores=[2, 3], state=state, job_manager=job_manager)
                
                # If there is a pending 3 we run it on 2 cores
                elif len(running_1c) == 1 and pending_3c:
                    self.update(running_1c[0], cores=[1], state=state, job_manager=job_manager)
                    self.run(pending_3c[0], cores=[2, 3], state=state, job_manager=job_manager)
                
                # If running 2 1 core job and we still have 1 core jobs pending, kick them off
                elif len(running_1c) == 2 and pending_1c:
                    self.run(pending_1c[0], cores=state.cores_available, state=state, job_manager=job_manager)

    def on_job_complete(self, completed_jobs, state: State, job_manager: JobManager) -> None:
        # If cores are available, simply schedule the highest core job possible
        num_cores_available = len(state.cores_available)

        # Look at what is running and upscale jobs if needed
        for bench in job_manager.running:
            cores_occupying = len(bench.get_cores())
            if cores_occupying < bench.cores_num and num_cores_available:
                # upscale but how?
                # state.relinquish_cores(bench.container)
                all_cores = sorted(bench.get_cores() + state.cores_available)
                # i dont think this is necessary if the rest is implemented correctly
                cores_to_occupy = min(len(all_cores), bench.cores_num)
                self.update(bench, cores=all_cores[-cores_to_occupy:], state=state, job_manager=job_manager)
        
        while bench := self._schedule_optimal_job(job_manager, num_cores_available):
            cores_to_occupy = min(num_cores_available, bench.cores_num)
            # print(f"NOW STARTING RUN: {bench.name}")
            self.run(bench, cores=state.cores_available[-cores_to_occupy:], state=state, job_manager=job_manager)

            # Update remaining cores
            num_cores_available -= cores_to_occupy
    

class ScalingStrategyPauseFerret(SchedulingStrategy):
    """No interference information, run without pausing containers by scaling/downscaling containers
    We make an exception to pause ferret because it does not seem to work well on high-load"""

    def __init__(self, ferret: Benchmark, ordering=None, colocations=None, logger=None):
        self.ferret = ferret
        super().__init__(ordering, colocations, logger)

    def _schedule_optimal_job(self, job_manager: JobManager, num_cores_available: int) -> Benchmark | None:
        """
        Schedule the next job in the pending list that satisfies the current state
        """
        next_bench_idx = 0
        core1_remaining = job_manager.get_pending_by_core(1)
        core2_remaining = job_manager.get_pending_by_core(2)
        core3_remaining = job_manager.get_pending_by_core(3)

        if num_cores_available == 3:        
            # Find the next benchmark in the ordering (pending list) that fits our current state
            # This is the optimal benchmark to run
            while next_bench_idx < len(job_manager.pending) and job_manager.pending[next_bench_idx].cores_num > num_cores_available:
                next_bench_idx += 1

            if next_bench_idx < len(job_manager.pending):
                return job_manager.pending[next_bench_idx]

        elif num_cores_available == 2:
            while next_bench_idx < len(job_manager.pending) and job_manager.pending[next_bench_idx].cores_num > num_cores_available:
                next_bench_idx += 1

            if next_bench_idx < len(job_manager.pending):
                return job_manager.pending[next_bench_idx]
            elif next_bench_idx >= len(job_manager.pending) and core3_remaining: 
                # Downscale a 3 if nothing below remains     
                return core3_remaining[0]

        elif num_cores_available == 1:
            if core1_remaining:
                return core1_remaining[0]
            elif core2_remaining:
                # Downscale a 2 if nothing below remains
                return core2_remaining[0]
            
            # We dont want to schedule a 3 core job on 1 core i think?
        
        return None

    def on_state_update(self, state: State, job_manager: JobManager) -> None:
        if state.load == Load.HIGH:
            running_3c = job_manager.get_running_by_core(3)
            if running_3c:
                if running_3c[0].name == "ferret":
                    # If we are running ferret we want to pause
                    self.pause(running_3c[0], state=state, job_manager=job_manager)
                    
                    # Unpause any jobs that are paused
                    to_unpause = []
                    for bench in job_manager.paused:
                        if bench.name != "ferret":
                            to_unpause.append(bench)
                    for bench in to_unpause:
                        self.unpause(bench, state=state, job_manager=job_manager)
                    else:
                        # If no jobs are paused, we want to spin up the next job?
                        self.on_job_complete([], state=state, job_manager=job_manager)

                else:
                    # Downscale the 3 core job to 2 cores
                    self.update(running_3c[0], cores=[2, 3], state=state, job_manager=job_manager)
                return
            
            running_2c = job_manager.get_running_by_core(2)
            running_1c = job_manager.get_running_by_core(1)
            if running_1c and running_2c:
                # If running a 2 and a 2, collocate the 1 core on the same cores as 2 core
                self.update(running_1c[0], cores=[2], state=state, job_manager=job_manager)
                return
            
            if len(running_1c) == 3:
                # If 3 1 core jobs were running...
                jobs_to_move = job_manager.get_jobs_on_core("1")
                self.update(jobs_to_move[0], cores=[2], state=state, job_manager=job_manager)
                
                return


            if len(running_2c) == 2:
                
                # If one of them is vips, we want to give it 2 cores
                first, second = running_2c
                vips = None
                other = None
                if first.name == "vips":
                    vips = first
                    other = second
                elif second.name == "vips":
                    vips = second
                    other = first

                if vips:
                    self.update(vips, cores=[2,3], state=state, job_manager=job_manager)
                    self.update(other, cores=[2], state=state, job_manager=job_manager)
                else:
                    # If two 2 core jobs were running (downscaled) we downscale further to a single core each 
                    self.update(first, cores=[2], state=state, job_manager=job_manager)
                    self.update(second, cores=[3], state=state, job_manager=job_manager)

        else:
            if self.ferret in job_manager.paused:
                # Pause all the current jobs
                to_pause = []
                for benchmark in job_manager.running:
                    to_pause.append(benchmark)
                for benchmark in to_pause:
                    self.pause(benchmark, state=state, job_manager=job_manager)
                
                # Unpause ferret
                self.unpause(self.ferret, state=state, job_manager=job_manager)

            running_3c = job_manager.get_running_by_core(3)
            if running_3c:
                # If running downscaled 3 core, upscale 
                self.update(running_3c[0], cores=[1, 2, 3], state=state, job_manager=job_manager)
                return

            running_2c = job_manager.get_running_by_core(2)
            running_3c = job_manager.get_running_by_core(3)
            running_1c = job_manager.get_running_by_core(1)
            pending_1c = job_manager.get_pending_by_core(1)
            pending_2c = job_manager.get_pending_by_core(2)
            pending_3c = job_manager.get_pending_by_core(3)
            if running_2c:
                # If the 2 core job had been downscaled (because 2 2-core running), we upscale again
                if len(running_2c) == 2:
                    self.update(running_2c[0], cores=[1, 2], state=state, job_manager=job_manager)
                elif running_1c:
                    # And running 1 core job, remove co-scheduling
                    self.update(running_1c[0], cores=[1], state=state, job_manager=job_manager)
                elif pending_1c:
                    # Or if 1 core jobs need running, kick them off
                    self.run(pending_1c[0], cores=[1], state=state, job_manager=job_manager)
                return
            
            if running_1c:

                if len(running_1c) == 3:
                    #TODO: maybe we can check the cores_available mapping instead
                    # If 3 1 core jobs are running, we now de-collocate the one running on core 2s
                    jobs_to_move = job_manager.get_jobs_on_core("2")

                    for bench in jobs_to_move:
                        # Move the one that was running on memcached core to safe core
                        self.update(bench, cores=[1], state=state, job_manager=job_manager)
                        break
                
                # If there is a pending 2, we run it on 2 cores
                elif len(running_1c) == 1 and pending_2c:
                    self.update(running_1c[0], cores=[1], state=state, job_manager=job_manager)
                    self.run(pending_2c[0], cores=[2, 3], state=state, job_manager=job_manager)
                
                # If there is a pending 3 we run it on 2 cores
                elif len(running_1c) == 1 and pending_3c:
                    self.update(running_1c[0], cores=[1], state=state, job_manager=job_manager)
                    self.run(pending_3c[0], cores=[2, 3], state=state, job_manager=job_manager)
                
                # If running 2 1 core job and we still have 1 core jobs pending, kick them off
                elif len(running_1c) == 2 and pending_1c:
                    self.run(pending_1c[0], cores=state.cores_available, state=state, job_manager=job_manager)

    def on_job_complete(self, completed_jobs, state: State, job_manager: JobManager) -> None:
        # If cores are available, simply schedule the highest core job possible
        num_cores_available = len(state.cores_available)

        # Look at what is running and upscale jobs if needed
        for bench in job_manager.running:
            cores_occupying = len(bench.get_cores())
            if cores_occupying < bench.cores_num and num_cores_available:
                # upscale but how?
                # state.relinquish_cores(bench.container)
                all_cores = sorted(bench.get_cores() + state.cores_available)
                num_cores_available = len(all_cores)
                # i dont think this is necessary if the rest is implemented correctly
                cores_to_occupy = min(num_cores_available, bench.cores_num)
                self.update(bench, cores=all_cores[-cores_to_occupy:], state=state, job_manager=job_manager)
        
        # If ferret is the job that just completed, unpause everything that is still paused
        if completed_jobs and completed_jobs[0] == self.ferret:
            to_unpause = []
            for bench in job_manager.paused:
                to_unpause.append(bench)
            for bench in to_unpause:
                self.unpause(bench, state=state, job_manager=job_manager)

        num_cores_available = len(state.cores_available)

        while bench := self._schedule_optimal_job(job_manager, num_cores_available):
            cores_to_occupy = min(num_cores_available, bench.cores_num)
            self.run(bench, cores=state.cores_available[-cores_to_occupy:], state=state, job_manager=job_manager)

            # Update remaining cores
            num_cores_available -= cores_to_occupy


class ShortestJobFirst(SchedulingStrategy):
    """No interference information, just runs jobs sequentially from shortest to longest"""
    pass

    # def on_state_update(self, state):
    #     return super().on_state_update(state)

    # def on_job_complete(self, state):
    #     return super().on_job_complete(state)


class ColocationStrategyV1(SchedulingStrategy):
    """Uses interference information, does not pause containers and does not allow colocation with memcached"""
    
    def __init__(self, ordering=None, colocations=None, logger=None):
        super().__init__(ordering, colocations, logger)

    def on_state_update(self, state, job_manager):
        return super().on_state_update(state, job_manager)
    
    def on_job_complete(self, completed_jobs, state, job_manager):
        return super().on_job_complete(completed_jobs, state, job_manager)

class ColocationStrategyV2(SchedulingStrategy):
    """Uses interference information, may pause containers and allows benchmarks to collocate with memcached"""
    pass

    # def on_state_update(self, state):
    #     return super().on_state_update(state)

    # def on_job_complete(self, state):
    #     return super().on_job_complete(state)
