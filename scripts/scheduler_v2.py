import argparse
import atexit
import signal
import subprocess
import sys
import time
from typing import List

import docker
import psutil
from strategies import SchedulingStrategy, IsolationStrategy, ScalingStrategy, ScalingStrategyPauseFerret
from utils.scheduler_logger import Job, SchedulerLogger
from utils.scheduler_utils import (
    Benchmark,
    JobManager,
    Load,
    MemcachedThresholds,
    State,
    update_config,
)


def signal_handler(*args, **kwargs):
    """Handle CTRL+C and other termination signals"""
    print("\nExiting gracefully...")

    # Reset memcached to run on a single core
    mc_pid = (
        subprocess.run("sudo pidof memcached", shell=True, capture_output=True)
        .stdout.decode()
        .strip()
    )
    subprocess.run(
        f"sudo taskset -a -cp 0-0 {mc_pid}",
        shell=True,
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
        self.logger = SchedulerLogger(question=question)
        self.cpu_thresholds = cpu_thresholds
        self.cores_used = {idx: False for idx in range(0, 4)}
        self.cores_used[0] = True
        self.state = State()

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

    def update_state(self):
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


        self.state.update_state(load_cores=cpu_perc_util, load=load, mc_utilization=mc_util)


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
        )

        # Log memcached initial run
        self.logger.job_start(
            Job.MEMCACHED,
            [str(core) for core in range(int(core_start), int(core_end) + 1)],
            threads,
        )

    def update_memcached(self) -> None:
        cores = ["0"]
        if self.state.load == Load.HIGH:
            cores.append("1")
            self.state.acquire_cores_manual([0, 1], "memcached")
        else:
            #! we are assuming that nothing will be coscheduled to memcached
            self.state.relinquish_cores_manual([1], "memcached")

        subprocess.run(
            f"sudo taskset -a -cp {cores[0]}-{cores[-1]} {self.mc_process.pid}",
            shell=True,
        )
        self.logger.update_cores(Job.MEMCACHED, cores)

    def run_strategy(self, strategy: SchedulingStrategy):
        self.job_manager.pending = strategy.ordering
        # first_benchmark = self.job_manager.get_highest_pending()
        self.update_state()
        
        strategy.on_job_complete([], self.state, self.job_manager)
        # strategy.run(first_benchmark, cores=sorted(
        #     [core for core in self.cores_used if not self.cores_used[core]]
        # )[-first_benchmark.cores_num:], state=self.state, job_manager=self.job_manager)

        tick = 0
        cooldown = 5
        start = time.perf_counter()
        while True:
            time.sleep(1)
            # Figure out if any jobs completed
            jobs_completed = self.job_manager.check_completed_jobs()            

            # Relinquish finished containers
            for benchmark in jobs_completed:
                self.logger.job_end(benchmark.to_job())
                self.state.relinquish_cores(benchmark.container)
            
            # If all jobs completed we terminate
            if self.job_manager.get_num_completed() == 7:
                break
        
            # Get the new state
            self.update_state()
            if tick % 5 == 0:
                print(self.job_manager)
                print(self.state)

            if jobs_completed:
                print(self.job_manager)
                print(self.state)
                strategy.on_job_complete(jobs_completed, self.state, self.job_manager)
            elif cooldown < 0 and self.state.load != self.current_load:
                cooldown = 5
                print(self.job_manager)
                print(self.state)
                self.update_memcached()  # Update memcached
                self.current_load = self.state.load
                strategy.on_state_update(self.state, self.job_manager)
            tick += 1
            cooldown -= 1


        end = time.perf_counter()
        makespan = (end - start) / 60

        self.logger.custom_event(
            Job.SCHEDULER, f"Total program makespan: {makespan:.4f} minutes"
        )
        print(f"Total program makespan: {makespan:.4f} minutes")

        self.job_manager.client.containers.prune()
        self.logger.end()


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

        # ShittyStrategy order
        # ORDER = [
        #     Benchmark("ferret", priority=1, thread_num=3, cores_num=3),
        #     Benchmark("freqmine", priority=2, thread_num=3, cores_num=3),
        #     Benchmark("canneal", priority=3, thread_num=2, cores_num=2),
        #     Benchmark("dedup", priority=3, thread_num=1, cores_num=1),
        #     Benchmark(
        #         "radix", priority=3, thread_num=4, cores_num=1, library="splash2x"
        #     ),
        #     Benchmark("blackscholes", priority=3, thread_num=3, cores_num=1),
        #     Benchmark("vips", priority=4, thread_num=3, cores_num=2),
        # ]

        # ScalingStrategy order

        # i think we want somthing similar to ferret for vips, 

        ORDER = [
            Benchmark("ferret", priority=1, thread_num=3, cores_num=3),
            Benchmark("freqmine", priority=2, thread_num=3, cores_num=3),
            # Canneal + Radix is not too bad
            #TODO: maybe we want canneal to 3 cores and blackscholes to 2?
            Benchmark("canneal", priority=3, thread_num=2, cores_num=2),
            Benchmark(
                "radix", priority=4, thread_num=4, cores_num=1, library="splash2x"
            ),

            # Vips first because 2 core, blackscholes first because it takes longer than dedup + more lenient
            Benchmark("blackscholes", priority=6, thread_num=3, cores_num=1),
            Benchmark("dedup", priority=7, thread_num=1, cores_num=1),
            Benchmark("vips", priority=5, thread_num=3, cores_num=2),
        ]
        strat = ScalingStrategy(
            ordering=ORDER, colocations=None, logger=controller.logger
        )

        # Execute the strategy
        controller.run_strategy(strat)
    else:
        controller.launch_memcached(threads=2, core_start="0", core_end="0")
        print("Launched Memcached!")


if __name__ == "__main__":
    main()
