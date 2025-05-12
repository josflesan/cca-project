import argparse
import subprocess
from typing import List

import docker
import psutil
from docker.models.containers import Container
from utils.scheduler_logger import Job, SchedulerLogger
from strategies import SchedulingStrategy, ShittyStrategy
from utils.scheduler_utils import (
    update_config,
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
import time
import sys
import signal
import atexit


def signal_handler(*args, **kwargs):
    """Handle CTRL+C and other termination signals"""
    print("\nExiting gracefully...")
    mc_pid = (
        subprocess.run("sudo pidof memcached", shell=True, capture_output=True)
        .stdout.decode()
        .strip()
        )

    client = docker.from_env()
    for container in client.containers.list():
        print(f"Killing Container: {container.name}")
        container.stop()

    client.containers.prune()
    sys.exit(0)


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
            subprocess.run("sudo pidof memcached", shell=True, capture_output=True)
            .stdout.decode()
            .strip()
        )
        print("mc_pid: ", mc_pid)
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

        print(f"MEMCACHED CORES: {mc_cores}")

        # Determine if high or low load based on thresholds
        load = self.current_load
        if mc_cores == 1 and mc_util >= self.cpu_thresholds.max_threshold:
            load = Load.HIGH
        elif mc_cores == 2 and mc_util < self.cpu_thresholds.min_threshold:
            load = Load.LOW
    
        # Get cores availabl
        cores_available = self._get_available_cores()

        # Return state
        return State(cpu_perc_util, load, mc_util, self.job_manager, cores_available)

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

    def launch_memcached(self, threads: int, core_start: str, core_end: str):
        # 1. Copy memcached config to the home directory and update it
        subprocess.run("cp /etc/memcached.conf ~/", shell=True, check=True)
        with open("/home/ubuntu/memcached.conf", "r") as f:
            config = f.read()

        updated_config = update_config(config, threads=threads)

        with open("/home/ubuntu/memcached.conf", "w") as f:
            f.write(updated_config)

        # 2. Move memcached config with sudo privileges
        subprocess.run(
            "sudo mv /home/ubuntu/memcached.conf /etc/memcached.conf", shell=True
        )

        # 3. Restart service
        subprocess.run("sudo systemctl restart memcached", shell=True)

        # 4. Pin memcached to cores
        subprocess.run(
            f"sudo taskset -a -cp {core_start}-{core_end} {self.mc_process.pid}",
            shell=True,
            capture_output=True,
        )

        # Log memcached initial run
        self.logger.job_start(
            Job.MEMCACHED,
            [str(core) for core in range(int(core_start), int(core_end) + 1)],
            threads,
        )

    def update_memcached(self, new_load: Load) -> None:
        cores = ["0"]
        if new_load == Load.HIGH:
            cores.append("1")
            self.cores_used[1] = True            
        else:
            #! we are assuming that nothing will be coscheduled to memcached
            self.cores_used[1] = False

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
                    print(f"NEXT BENCHMARK: {benchmark}")
                    print(f"NEXT CORES: {cores}")
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
        first_benchmark = self.job_manager.get_highest_pending()
        self._handle_run(
            first_benchmark,
            cores=sorted(
                [core for core in self.cores_used if not self.cores_used[core]][
                    : first_benchmark.cores_num
                ]
            ),
        )

        tick = 0
        while True:
            time.sleep(1)    
                    
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

            if tick % 5 == 0:
                print(self.job_manager)
                print(state)

            if jobs_completed:
                print(state)
                strategy.on_job_complete(jobs_completed, state)
                self.flush_buffer(strategy)
            elif state.load != self.current_load:
                print(state)
                self.update_memcached(state.load)  # Update memcached
                self.current_load = state.load
                strategy.on_state_update(state)
                self.flush_buffer(strategy)

            tick += 1


def main():
    atexit.register(signal_handler)
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    signal.signal(signal.SIGABRT, signal_handler)  # Crash


    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    # Initialize the controller
    thresholds = MemcachedThresholds()
    controller = Controller(question=3, cpu_thresholds=thresholds)
    
    if args.run:
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
    else:
        controller.launch_memcached(
            threads=2,
            core_start="0",
            core_end="0"
        )
        print("Launched Memcached!")


if __name__ == "__main__":
    main()
