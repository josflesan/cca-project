import re
from dataclasses import dataclass, field
from typing import List, Collection
from docker.models.containers import Container
from .scheduler_logger import Job
import enum
from enum import Enum


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

    def is_paused(self):
        # Maybe we have to reload, dont think so tho
        return self.container and self.container.status == "paused"

    def __str__(self):
        return f"(name={self.name}, p={self.priority})"
    
    def __repr__(self):
        return f"(name={self.name}, p={self.priority})"


@dataclass
class JobManager:
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

    ############# GETTERS #############
    def get_running_by_core(self, cores: int) -> List[Benchmark]:
        result = [
            benchmark for benchmark in self.running if benchmark.cores_num == cores
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


@dataclass(slots=True)
class State:
    load_cores: List[float]
    load: Load
    mc_utilization: float
    job_manager: JobManager
    cores_available: List[int]

    def __str__(self):
        ret_str = "--------------- STATE ---------------\n"
        ret_str += f"MC Util:\t {self.mc_utilization}\n"
        ret_str += f"LOAD:\t {self.load}\n"
        ret_str += f"CORES AVAILABLE:\t {self.cores_available}\n"

        return ret_str


@dataclass
class MemcachedThresholds:
    min_threshold: float = 80.0
    max_threshold: float = 90.0


################# COMMANDS #################
class Command:
    pass


@dataclass(frozen=True, slots=True)
class Run(Command):
    benchmark: Benchmark
    cores: List[int]


@dataclass(frozen=True, slots=True)
class Update(Command):
    benchmark: Benchmark
    cores: List[int]


@dataclass(frozen=True, slots=True)
class Pause(Command):
    benchmark: Benchmark


@dataclass(frozen=True, slots=True)
class Unpause(Command):
    benchmark: Benchmark
