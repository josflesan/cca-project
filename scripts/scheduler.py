import docker
import heapq
import argparse
import re
import subprocess
import dataclasses
from typing import List
from scheduler_logger import SchedulerLogger, Job
import psutil
from collections import deque, defaultdict
from typing import Dict, Deque
from docker.models.containers import Container
import time
import sys
import signal
import atexit


def signal_handler(*args, **kwargs):
    """Handle CTRL+C and other termination signals"""
    print("\nExiting gracefully...")
    client = docker.from_env()
    for container in client.containers.list():
        print(f"Killing Container: {container.name}")
        container.stop()

    client.containers.prune()
    sys.exit(0)

@dataclasses.dataclass
class Benchmark:
    name: str
    priority: int
    thread_num: int = 1
    cores_num: int = 1
    library: str = "parsec"

    def __post_init__(self):
        if self.name == "radix":
            self.image: str = "anakli/cca:splash2x_radix"
            self.library: str = "splash2x"
        else:
            self.image: str = f"anakli/cca:parsec_{self.name}"

    def to_job(self) -> Job:
        return Job._member_map_[self.name.upper()]




@dataclasses.dataclass
class MemcachedConfig:
    memcached_threads: int
    memcached_cores: List[str]
    cpu_max_threshold: float = 60.0
    cpu_min_threshold: float = 60.0

ORDER = [
    Benchmark("ferret", priority=1, thread_num=3, cores_num=3),
    Benchmark("freqmine", priority=2, thread_num=3, cores_num=3),
    Benchmark("canneal", priority=3, thread_num=2, cores_num=2),
    Benchmark("dedup", priority=3, thread_num=1, cores_num=1),
    Benchmark("radix", priority=3, thread_num=4, cores_num=1),
    Benchmark("blackscholes", priority=3, thread_num=3, cores_num=1),
    Benchmark("vips", priority=4, thread_num=3, cores_num=2),
]


def update_config(config: str, internal_ip: str | None = None, threads: int = 1) -> str:
    # Replace memory limit (-m) with 1024
    config = re.sub(r"^-m\s+\d+", "-m 1024", config, flags=re.MULTILINE)
    # Replace listen address (-l) with internal IP
    if internal_ip:
        config = re.sub(r"^-l\s+\S+", f"-l {internal_ip}", config, flags=re.MULTILINE)
    # Set number of threads (-t); add if not present
    if re.search(r"^-t\s+\d+", config, flags=re.MULTILINE):
        config = re.sub(r"^-t\s+\d+", f"-t {threads}", config, flags=re.MULTILINE)
    else:
        config += f"\n-t {threads}\n"

    return config


class Controller:
    def __init__(self, question: int, memcached_config: MemcachedConfig):
        # Initialize client and logger
        self.client = self._init_client()
        self.logger = SchedulerLogger(question=question)
        self.pid = None  # Memcached PID
        self.benchmark_idx = 0  # ID of the next benchmark to start according to ORDER
        self.memcached_config = memcached_config
        self.cores_used = {
            idx: False for idx in range(0, 4)
        }
        self.cores_used[0] = True

        # Get PID of memcached process
        self.pid = subprocess.run(
            "pidof memcached", shell=True, capture_output=True
        ).stdout.decode().strip()
        self.mc_process = psutil.Process(int(self.pid))
        
        # Queues
        self.pending: Dict[int, Deque[Benchmark]] = {
            1: deque([b for b in ORDER if b.cores_num == 1]),
            2: deque([b for b in ORDER if b.cores_num == 2]),
            3: deque([b for b in ORDER if b.cores_num == 3]),
        }
        self.running: Dict[int, Deque[Container]] = defaultdict(deque)
        self.paused: Dict[int, Deque[Container]] = defaultdict(deque)
        self.completed = []

    def _init_client(self):
        # Give docker sudo permissions
        # subprocess.run("sudo usermod -a -G docker jfleitas", text=True, shell=True)

        # Get Docker client SDK
        return docker.from_env()

    def _get_cpu_utilization(self, interval: int = 1) -> List[float]:
        per_cpu_percent = psutil.cpu_percent(interval=interval, percpu=True)
        return per_cpu_percent
    
    def _reload_containers(self):
        for containers in self.running.values():
            for container in containers:
                container.reload()

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
            f"sudo taskset -a -cp {core_start}-{core_end} {self.pid}",
            shell=True,
            capture_output=True,
        )

        # Log memcached initial run
        self.logger.job_start(
            Job.MEMCACHED,
            [str(core) for core in range(int(core_start), int(core_end) + 1)],
            threads,
        )

    def print_running_containers(self):
        print("Running containers:")
        for thread, containers in self.running.items():
            print(f"{thread}: ", [container.name for container in containers])

    def schedule_next_benchmark(self, cores: int, cores_to_pin: List[str]) -> None:
        benchmark = self.pending[
            cores
        ].popleft()  # Remove pending job from the front of the queue
        print(f"now scheduling Benchmark: {benchmark}")
        container = self.client.containers.run(
            image=benchmark.image,
            command=f"./run -a run -S {benchmark.library} -p {benchmark.name} -i native -n {benchmark.thread_num}",
            name=benchmark.name,  # TODO: might need to change this
            cpuset_cpus=f"{cores_to_pin[0]}-{cores_to_pin[-1]}",
            detach=True,
            remove=False,
        )

        # Set the cores used to True
        for core in cores_to_pin:
            self.cores_used[int(core)] = True

        self.logger.job_start(
            benchmark.to_job(),
            initial_cores=cores_to_pin,
            initial_threads=benchmark.thread_num,
        )

        self.running[cores].append(container)

    def pause_container(self, cores: int):
        container = self.running[cores].popleft()
        container.reload()
        if container.status != "exited":
            self.relinquish_cores(container)
            container.pause()
            self.paused[cores].append(container)
            self.logger.job_pause(Job._member_map_[container.name.upper()])
        else:
            self.clean_up_containers()

    def unpause_container(self, cores: int, cores_to_pin: List[str]):
        container = self.paused[cores].popleft()
        container.reload()
        if container.status != "exited":
            container.unpause()
            
            # Update the number of cores
            self.update_cores(container, cores_to_pin)

            self.running[cores].append(container)
            self.logger.job_unpause(Job._member_map_[container.name.upper()])
        else:
            # maybe this is wrong, need to think more about this
            self.clean_up_containers()

    def update_cores(self, container: Container, cores_to_pin: List[str]):
        # Set the previous cores to False
        container.update(cpuset_cpus=f"{cores_to_pin[0]}-{cores_to_pin[-1]}")

        # Set the new cores to True
        for core in cores_to_pin:
            self.cores_used[int(core)] = True

        self.logger.update_cores(Job._member_map_[container.name.upper()], cores_to_pin)

    def get_num_cores_available(self) -> int:
        return 4 - sum(self.cores_used.values())  # Subtract cores used from cores available in VM

    def get_available_cores(self) -> List[str]:
        cores_available = []
        for core, is_avail in self.cores_used.items():
            if not is_avail:
                cores_available.append(str(core))
        return sorted(cores_available)


    def update_memcached(self, cores: List[str]):
        print(f"MEMCACHED PID: {self.pid}")
        subprocess.run(
            f"sudo taskset -a -cp {cores[0]}-{cores[-1]} {self.pid}",
            shell=True
        )
        self.logger.update_cores(Job.MEMCACHED, cores)

    def relinquish_cores(self, container: Container) -> None:
        core_start, core_end = container.attrs['HostConfig']['CpusetCpus'].split("-")
        for core in range(int(core_start), int(core_end) + 1):
            print(f"Relinquishing core {core}...")
            self.cores_used[core] = False

    def clean_up_containers(self):
        self._reload_containers()
        # Update the completed and running queues if job is completed
        todelete = []
        for threads, containers in self.running.items():
            for container in containers:
                if container.status == "exited":
                    print(f"{container.name} finished, now freeing cores...")
                    self.relinquish_cores(container)

                    todelete.append((threads, container))
                    self.completed.append(container.name)
        
                    # Log job end
                    self.logger.job_end(Job._member_map_[container.name.upper()])

        # Cleanup the containers
        for threads, container in todelete:
            self.running[threads].remove(container)


    def run_loop(self):
        not_completed = True
        cycle = -1
        self.memcached_cores = 1  # Number of cores used by memcached

        start = time.perf_counter()
        not_transition = True
        # Main Event Loop
        while not_completed:
            if not_transition:
                time.sleep(1)
            cycle += 1

            # Reload containers and check for all completions
            self.clean_up_containers()
                
            # If we have completed all of the benchmarks, terminate
            if len(self.completed) == 7:
                not_completed = False

            if cycle % 50 == 0:
                self.print_running_containers()

            # Get CPU info
            cpu_perc_util = self._get_cpu_utilization()
            memcache_utilization = self.mc_process.cpu_percent()

            # Get available cores
            num_cores_available = self.get_num_cores_available()
            cores_available = self.get_available_cores()

            print(f"CORES USED: {self.cores_used}")
            print(f"Number of Cores Available: {num_cores_available}")
            print(f"Cores Available: {cores_available}")
            print(f"CPU Core 0 Util: {cpu_perc_util[0]}")
            print(f"CPU Core 1 Util: {cpu_perc_util[1]}")
            
            print(f"CPU PROCESS Util: {memcache_utilization}")
            
            print(f"LOAD: {'high' if self.memcached_cores == 2 else 'low'}")
            print(f"MC running on:  {self.mc_process.cpu_affinity()}")


            if self.memcached_cores == 1:

                # Check if max threshold met
                if memcache_utilization >= self.memcached_config.cpu_max_threshold:
                    
                    if self.running[3]:
                        self.pause_container(cores=3)
                    if self.running[2]:
                        self.pause_container(cores=2)
                    if self.running[1]:
                        self.pause_container(cores=1)

                    self.update_memcached(cores=["0", "1"])
                    self.memcached_cores = 2
                    self.cores_used[1] = True
                    print("HIT MAX THRESHOLD!")
                    not_transition=False
                    continue
                else:
                    not_transition=True

                # Check available cores and run highest in paused, fall back to pending
                if self.paused[3] and num_cores_available == 3:
                    self.unpause_container(cores=3, cores_to_pin=cores_available)
                elif self.pending[3] and num_cores_available == 3:
                    self.schedule_next_benchmark(cores=num_cores_available, cores_to_pin=cores_available)
                elif self.paused[2] and num_cores_available >= 2:
                    self.unpause_container(cores=2, cores_to_pin=cores_available[-2:])
                elif self.pending[2] and num_cores_available >= 2:
                    self.schedule_next_benchmark(cores=2, cores_to_pin=cores_available[-2:])
                elif self.paused[1] and num_cores_available >= 1:
                    self.unpause_container(cores=1, cores_to_pin=[cores_available[-1]])
                elif self.pending[1] and num_cores_available >= 1:
                    self.schedule_next_benchmark(cores=1, cores_to_pin=[cores_available[-1]])

            elif self.memcached_cores == 2:
                # Check if min threshold met
                if memcache_utilization < self.memcached_config.cpu_min_threshold:
                    if self.running[2]:
                        self.pause_container(cores=2)
                    if self.running[3]:
                        self.pause_container(cores=3)

                    self.update_memcached(cores=["0"])
                    self.memcached_cores = 1
                    self.cores_used[1] = False
                    print("HIT MIN THRESHOLD!")
                    not_transition=False
                    self._reload_containers()
                    continue
                else:
                    not_transition=True
                
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

        end = time.perf_counter()
        makespan = (end - start) / 60

        self.logger.custom_event(
            Job.SCHEDULER, f"Total program makespan: {makespan:.4f} minutes"
        )
        print(f"Total program makespan: {makespan:.4f} minutes")

        self.client.containers.prune()

        # After loop is finished executing, end logger
        self.logger.end()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    atexit.register(signal_handler)
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    signal.signal(signal.SIGABRT, signal_handler)  # Crash

    mc_config = MemcachedConfig(
        cpu_max_threshold=90.0,
        cpu_min_threshold=80.0,
        memcached_threads=2,
        memcached_cores=["0"],
    )
    controller = Controller(question=2, memcached_config=mc_config)

    if args.run:
        controller.run_loop()
    else:
        controller.launch_memcached(
            mc_config.memcached_threads,
            mc_config.memcached_cores[0],
            mc_config.memcached_cores[-1],
        )
        print("Launched Memcached!")


if __name__ == "__main__":
    main()

# QUESTIONS:
#  WHEN TO PAUSE : When cpu utilization for memcached gets too high
#  WHEN TO START A JOB: When at the top of the queue and some cores are available, We only want to pause if its a single threaded program
#  WHEN TO UNPAUSE: when either cpu utilizations starts getting lower again, or
#  HOW TO KEEP TRACK OF OUR BENCHMARKS/CONTAINERS AND CORES

# ideas:
# MAYBE CO-SCHEDULE RADIX AND CANNEAL?
# do an experiment to see the performance differences of: 
# 3 threads on 3 cores, 
# 3 threads on 2 cores,
# 2 threads and 2 cores 
