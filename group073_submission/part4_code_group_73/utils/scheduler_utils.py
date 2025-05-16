import re
from dataclasses import dataclass, field
from typing import List, Dict, Set
from docker.models.containers import Container
from .scheduler_logger import Job
import enum
from enum import Enum
from collections import defaultdict
import docker


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


class Load(Enum):
    LOW = enum.auto()
    HIGH = enum.auto()


@dataclass
class Benchmark:
    name: str
    priority: int = 0
    thread_num: int = 1
    cores_num: int = 1
    container: Container | None = None
    library: str = "parsec"

    def __post_init__(self):
        if self.name == "radix":
            self.image: str = "anakli/cca:splash2x_radix"
            self.library: str = "splash2x"
        else:
            self.image: str = f"anakli/cca:parsec_{self.name}"

    def to_job(self) -> Job:
        return Job._member_map_[self.name.upper()]

    def attach_container(self, container: Container) -> None:
        self.container = container

    def get_cores(self) -> List[int]:
        core_start, core_end = self.container.attrs['HostConfig']['CpusetCpus'].split("-")
        return list(range(int(core_start), int(core_end) + 1))

    def is_paused(self):
        # Maybe we have to reload, dont think so tho
        return self.container and self.container.status == "paused"

    def __str__(self):
        return f"(name={self.name}, p={self.priority})"
    
    def __repr__(self):
        return f"(name={self.name}, p={self.priority})"

@dataclass(slots=True)
class State:
    load_cores: List[float] = field(default_factory=list)
    load: Load = Load.LOW
    mc_utilization: float = 0.0
    cores_available: List[int] = field(default_factory=list)
    cores_used: Dict[int, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def __post_init__(self):
        self.cores_used = {i: set() for i in range(4)}
        self.cores_used[0].add("memcached")
        
        # At the beginning we have 3 cores available
        self.cores_available = [1, 2, 3]

    def update_state(self, **kwargs):
        self.load_cores = kwargs.get("load_cores", self.load)
        self.load = kwargs.get("load", self.load)
        self.mc_utilization = kwargs.get("mc_utilization", self.mc_utilization)

    def acquire_cores(self, container: Container):
        # TODO: maybe delete if this doesnt fix
        container.reload()
        core_start, core_end = container.attrs["HostConfig"]["CpusetCpus"].split("-")
        for core in range(int(core_start), int(core_end) + 1):
            self.cores_used[core].add(container.name)
        
        # Update cores available
        self.cores_available = sorted([core for core, used in self.cores_used.items() if not used])

    def acquire_cores_manual(self, cores: List[int], job: str):
        for core in cores:
            if job not in self.cores_used[core]:
                self.cores_used[core].add(job)
        
        # Update cores available
        self.cores_available = sorted([core for core, used in self.cores_used.items() if not used])

    def relinquish_cores(self, container: Container):
        core_start, core_end = container.attrs["HostConfig"]["CpusetCpus"].split("-")
        for core in range(int(core_start), int(core_end) + 1):
            self.cores_used[core].remove(container.name)
        
        # Update cores available
        self.cores_available = sorted([core for core, used in self.cores_used.items() if not used])

    def relinquish_cores_manual(self, cores: List[int], job: str):
        for core in cores:
            self.cores_used[core].remove(job)

        # Update cores available
        self.cores_available = sorted([core for core, used in self.cores_used.items() if not used])

    def __str__(self):
        ret_str = "--------------- STATE ---------------\n"
        ret_str += f"MC Util:\t {self.mc_utilization}\n"
        ret_str += f"LOAD:\t {self.load}\n"
        ret_str += f"CORES AVAILABLE:\t {self.cores_available}\n"
        ret_str += f"CORES USED:\t {self.cores_used}\n"

        return ret_str

@dataclass
class JobManager:
    client: docker.DockerClient = field(default_factory=docker.from_env)
    completed: int = 0
    running: List[Benchmark] = field(default_factory=list)
    paused: List[Benchmark] = field(default_factory=list)
    pending: List[Benchmark] = field(default_factory=list)

    ############# METHODS #############
    def check_completed_jobs(self):
        completed_jobs = []
        for benchmark in self.running:
            benchmark.container.reload()
            if benchmark.container.status == "exited":
                completed_jobs.append(benchmark)
                self.completed += 1

        # Remove benchmark from running
        for benchmark in completed_jobs:
            self.running.remove(benchmark)

        return completed_jobs
    
    def update(self, bench: Benchmark, cores: List[int], state: State) -> bool:
        container = bench.container
        container.reload()

        # Relinquish previous cores for this job
        if bench in self.running:
            state.relinquish_cores(container)

        # Update the cores used by the container
        container.update(cpuset_cpus=f"{cores[0]}-{cores[-1]}")

        # Set the new cores to be used
        if bench in self.running:
            state.acquire_cores(container)

    def pause(self, bench: Benchmark, state: State) -> bool:
        container = bench.container
        container.reload()
        if container.status != "exited":
            state.relinquish_cores(container)
            container.pause()

            self.paused.append(bench)
            self.running.remove(bench)

            return True
        
        return False

    def unpause(self, bench: Benchmark, state: "State") -> bool:
        container = bench.container
        container.reload()

        if container.status != "exited":
            container.unpause()
            # Update paused and running queues
            self.paused.remove(bench)
            self.running.append(bench)

            # Add the new cores
            state.acquire_cores(container)

            return True
        
        return False

    def run(self, bench: Benchmark, cores: List[int], state: State) -> None:
        container = self.client.containers.run(
            image=bench.image,
            command=f"./run -a run -S {bench.library} -p {bench.name} -i native -n {bench.thread_num}",
            name=bench.name,
            cpuset_cpus=f"{cores[0]}-{cores[-1]}",
            detach=True,
            remove=False,
        )
        
        bench.attach_container(container)  # Attach the container to the bench

        # Set the cores used to True
        state.acquire_cores(container)

        # Add job to the running queue
        self.running.append(bench)
        self.pending.remove(bench)


    ############# GETTERS #############
    def get_jobs_on_core(self, core: str) -> List[Benchmark]:
        relevant = []
        for benchmark in self.running:
            cores = benchmark.container.attrs["HostConfig"]["CpusetCpus"].split("-")

            # Move the one that was running on memcached core to safe core
            if str(core) in cores:
                relevant.append(benchmark)

        return relevant
    
    def get_running_by_core(self, cores: int) -> List[Benchmark]:
        result = [
            bench for bench in self.running if bench.cores_num == cores
        ]
        return result

    def get_paused_by_core(self, cores: int) -> List[Benchmark]:
        result = [
            benchmark for benchmark in self.paused if benchmark.cores_num == cores
        ]
        return result

    def get_pending_by_core(self, cores: int) -> List[Benchmark]:
        result = [
            benchmark for benchmark in self.pending if benchmark.cores_num == cores
        ]
        return result

    def get_num_completed(self) -> int:
        return self.completed

    def get_num_running(self) -> int:
        return len(self.running)

    def get_num_paused(self) -> int:
        return len(self.paused)

    def get_num_pending(self) -> int:
        return len(self.pending)

    def get_non_running(self) -> list:
        return self.paused + self.pending

    def get_highest_pending(self) -> Benchmark:
        return min(self.pending, key=lambda b: b.priority)

    ############# MAGIC METHODS #############

    def __str__(self):
        ret_str = "--------------- JOB MANAGER ---------------\n"
        ret_str += f"COMPLETED:\t {self.completed}\n"
        ret_str += f"RUNNING:\t {self.running}\n"
        ret_str += f"PAUSED:\t {self.paused}\n"
        ret_str += f"PENDING:\t {self.pending}\n"

        return ret_str


@dataclass
class MemcachedThresholds:
    min_threshold: float = 90.0
    max_threshold: float = 80.0  # TODO: is this good?
