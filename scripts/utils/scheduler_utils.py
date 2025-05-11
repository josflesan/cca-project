from dataclasses import dataclass, field
from typing import List, Collection
from docker.models.containers import Container
from ..scheduler_logger import SchedulerLogger, Job
import enum
from enum import Enum

class Load(Enum):
    LOW = enum.auto(),
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

    
@dataclass
class JobManager:
    queue_types: List[type]
    completed: int = 0
    running: List[Benchmark] = field(default_factory=list)
    paused: List[Benchmark] = field(default_factory=list)
    pending: List[Benchmark] = field(default_factory=list)

    def __post_init__(self):
        self.running = field(default_factory=self.queue_types[0])
        self.paused = field(default_factory=self.queue_types[1])
        self.pending = field(default_factory=self.queue_types[2])

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
        relevant_bench = []
        for bench in self.running:
            if bench.cores_num == cores:
                relevant_bench.append(bench)

        return relevant_bench
    
    def get_paused_by_core(self, cores: int) -> List[Benchmark]:
        relevant_bench = []
        for bench in self.paused:
            if bench.cores_num == cores:
                relevant_bench.append(bench)
        
        return relevant_bench

    def get_pending_by_core(self, cores: int) -> List[Benchmark]:
        relevant_bench = []
        for bench in self.pending:
            if bench.cores_num == cores:
                relevant_bench.append(bench)
        
        return relevant_bench

    def get_num_completed(self) -> int:
        return self.completed

    def get_num_running(self) -> int:
        return len(self.running)
    
    def get_num_paused(self) -> int:
        return len(self.paused)
    
    def get_num_pending(self) -> int:
        return len(self.pending)

@dataclass(slots=True)
class State:
    load_cores: List[float]
    load_enum: Load
    job_queues: JobManager
    cores_available: List[int]

    def get_cores_available(self) -> List[int]:
        return self.cores_available
    
    def get_num_cores_available(self) -> int:
        return len(self.cores_available)
    
@dataclass
class MemcachedThresholds:
    min_threshold: float = 80.0
    max_threshold: float = 90.0


################# COMMANDS #################

@dataclass
class Run:
   type: str
   bench: Benchmark
   cores: List[int]

   def __post_init__(self):
       self.type = self.bench.name

@dataclass
class Update:
   type: str
   bench: Benchmark
   cores: List[int]

   def __post_init__(self):
       self.type = self.bench.name

@dataclass
class Pause:
   type: str
   bench: Benchmark

   def __post_init__(self):
       self.type = self.bench.name

@dataclass
class Unpause:
   type: str
   bench: Benchmark

   def __post_init__(self):
       self.type = self.bench.name




