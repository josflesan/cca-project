import subprocess
from typing import List

import docker
import psutil
from docker.models.containers import Container
from scheduler_logger import Job, SchedulerLogger
from strategies import SchedulingStrategy, ShittyStrategy
from utils.scheduler_utils import (
    Benchmark,
    JobManager,
    Load,
    MemcachedThresholds,
    Pause,
    Run,
    State,
    Unpause,
    Update,
)


class Controller:
    def __init__(self, question: int, cpu_thresholds: MemcachedThresholds):
        # Initialize client and logger
        self.client = self._init_client()
        self.logger = SchedulerLogger(question=question)
        self.cpu_thresholds = cpu_thresholds
        self.cores_used = {idx: False for idx in range(0, 4)}
        self.cores_used[0] = True
        # Get PID of memcached process and save process
        mc_pid = (
            subprocess.run("pidof memcached", shell=True, capture_output=True)
            .stdout.decode()
            .strip()
        )
        self.mc_process = psutil.Process(int(mc_pid))
        # Initial load is LOW
        self.current_load = Load.LOW
        # Get the job manager
        self.job_manager = JobManager()

    def _init_client(self):
        # Give docker sudo permissions
        # subprocess.run("sudo usermod -a -G docker jfleitas", text=True, shell=True)

        # Get Docker client SDK
        return docker.from_env()

    def _get_cpu_utilization(self, interval: int = 1) -> List[float]:
        per_cpu_percent = psutil.cpu_percent(interval=interval, percpu=True)
        return per_cpu_percent

    def _get_available_cores(self):
        cores_available = []
        for core, is_avail in self.cores_used.items():
            if not is_avail:
                cores_available.append(str(core))
        return sorted(cores_available)

    def _get_num_cores_available(self):
        return 4 - sum(
            self.cores_used.values()
        )  # Subtract cores used from cores available in VM

    def _get_state(self):
        # Compute the load
        cpu_perc_util = self._get_cpu_utilization()
        mc_util = self.mc_process.cpu_percent()
        mc_cores = len(self.mc_process.cpu_affinity())

        # Determine if high or low load based on thresholds
        load = self.current_load
        if mc_cores == 1 and mc_util >= self.cpu_thresholds.max_threshold:
            load = Load.HIGH
        elif mc_cores == 2 and mc_util < self.cpu_thresholds.min_threshold:
            load = Load.LOW

        # Get cores available
        cores_available = self._get_available_cores()

        # Return state
        return State(cpu_perc_util, load, self.job_manager, cores_available)

    def _handle_run(self, benchmark: Benchmark, cores: List[int]) -> None:
        job_manager = self.job_manager

        print(f"Now scheduling benchmark: {benchmark.name}")

        container = self.client.containers.run(
            image=benchmark.image,
            command=f"./run -a run -S {benchmark.library} -p {benchmark.name} -i native -n {benchmark.thread_num}",
            name=benchmark.name,
            cpuset_cpus=f"{cores[0]}-{cores[-1]}",
            detach=True,
            remove=False,
        )
        benchmark.attach_container(container)  # Attach the container to the benchmark

        # Set the cores used to True
        for core in cores:
            self.cores_used[core] = True

        self.logger.job_start(
            benchmark.to_job(),
            initial_cores=cores,
            initial_threads=benchmark.thread_num,
        )

        # Add job to the running queue
        job_manager.running.append(benchmark)
        job_manager.pending.remove(benchmark)

    def _handle_update(self, benchmark: Benchmark, cores: List[int]) -> None:
        container = benchmark.container

        # Relinquish the previous cores
        self.relinquish_cores(container)

        # Update the cores used by the container
        container.update(cpuset_cpus=f"{cores[0]}-{cores[-1]}")

        # Set the new cores to True
        for core in cores:
            self.cores_used[core] = True

        self.logger.update_cores(Job._member_map_[container.name.upper()], cores)

    def _handle_pause(self, benchmark: Benchmark):
        job_manager = self.job_manager

        container = benchmark.container
        container.reload()

        if container.status != "exited":
            self.relinquish_cores(container)
            container.pause()

            job_manager.paused.append(benchmark)
            job_manager.running.remove(benchmark)
            self.logger.job_pause(Job._member_map_[benchmark.name.upper()])

    # might wanna figure out if we wanna pass in cores here?
    def _handle_unpause(self, benchmark: Benchmark) -> None:
        job_manager = self.job_manager

        container = benchmark.container
        container.reload()

        if container.status != "exited":
            container.unpause()
            # Update paused and running queues
            job_manager.paused.remove(benchmark)
            job_manager.running.append(benchmark)
            self.logger.job_unpause(Job._member_map_[benchmark.name.upper()])

    def relinquish_cores(self, container: Container) -> None:
        core_start, core_end = container.attrs["HostConfig"]["CpusetCpus"].split("-")
        for core in range(int(core_start), int(core_end) + 1):
            print(f"Relinquishing core {core}...")
            self.cores_used[core] = False

    def update_memcached(self, new_load: Load) -> None:
        cores = ["0"]
        if new_load == Load.HIGH:
            cores.append("1")

        subprocess.run(
            f"sudo taskset -a -cp {cores[0]}-{cores[-1]} {self.mc_process.pid}",
            shell=True,
        )
        self.logger.update_cores(Job.MEMCACHED, cores)

    def flush_buffer(self, strategy: SchedulingStrategy) -> None:
        """
        Flushes the buffer by executing every command and reseting the buffer
        """
        for command in strategy.command_buffer:
            match command:
                case Run(benchmark, cores):
                    self._handle_run(benchmark, cores)
                case Pause(benchmark):
                    self._handle_pause(benchmark)
                case Unpause(benchmark):
                    self._handle_unpause(benchmark)
                case Update(benchmark, cores):
                    self._handle_update(benchmark, cores)
                case _:
                    raise RuntimeError("Unrecognised Command passed")
        strategy.command_buffer = []

    def run_strategy(self, strategy: SchedulingStrategy):
        self.job_manager.pending = strategy.ordering
        first_benchmark = max(self.job_manager.pending, lambda b: b.priority)
        self._handle_run(
            first_benchmark,
            cores=sorted(
                [core for core in self.cores_used if not self.cores_used[core]][
                    : first_benchmark.cores_num
                ]
            ),
        )

        while True:
            print(self.job_manager)

            # Figure out if any jobs completed
            jobs_completed = self.job_manager.check_completed_jobs()

            # If all jobs completed we terminate
            if self.job_manager.get_num_completed() == 7:
                break

            # Relinquish finished containers
            for benchmark in jobs_completed:
                self.relinquish_cores(benchmark.container)

            # Get the new state
            state = self._get_state()
            print(state)
            if jobs_completed:
                strategy.on_job_complete(jobs_completed, state)
                self.flush_buffer(strategy)
            elif state.load != self.current_load:
                self.update_memcached(state.load)  # Update memcached
                self.current_load = state.load
                strategy.on_state_update(state)
                self.flush_buffer(strategy)


def main():
    # Initialize the controller
    thresholds = MemcachedThresholds()
    controller = Controller(question=3, cpu_thresholds=thresholds)

    # Initialize ordering and strategy
    ORDER = [
        Benchmark("ferret", priority=1, thread_num=3, cores_num=3),
        Benchmark("freqmine", priority=2, thread_num=3, cores_num=3),
        Benchmark("canneal", priority=3, thread_num=2, cores_num=2),
        Benchmark("dedup", priority=3, thread_num=1, cores_num=1),
        Benchmark("radix", priority=3, thread_num=4, cores_num=1, library="splash2x"),
        Benchmark("blackscholes", priority=3, thread_num=3, cores_num=1),
        Benchmark("vips", priority=4, thread_num=3, cores_num=2),
    ]
    strat = ShittyStrategy(ordering=ORDER, colocations=None, logger=controller.logger)
    # Execute the strategy
    controller.run_strategy(strat)


if __name__ == "__main__":
    main()
