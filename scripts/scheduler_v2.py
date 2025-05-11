import docker
import argparse
import subprocess
import psutil
from typing import List
from docker.models.containers import Container

from strategies import SchedulingStrategy
from scheduler_logger import SchedulerLogger, Job
from utils.scheduler_utils import MemcachedThresholds, Benchmark, Load, JobManager, State, Run, Update, Pause, Unpause

class Controller:
    def __init__(self, question: int, cpu_thresholds: MemcachedThresholds):
        
        # Initialize client and logger
        self.client = self._init_client()
        self.logger = SchedulerLogger(question=question)
        self.cpu_thresholds = cpu_thresholds
        self.cores_used = {
            idx: False for idx in range(0, 4)
        }
        self.cores_used[0] = True
        
        # Get PID of memcached process and save process
        mc_pid = subprocess.run(
            "pidof memcached", shell=True, capture_output=True
        ).stdout.decode().strip()
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
        return 4 - sum(self.cores_used.values())  # Subtract cores used from cores available in VM

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
    
    def _handle_run(self, strategy: SchedulingStrategy, bench: Benchmark, cores: List[int]):
        job_manager = strategy.job_manager
        print(f"Now scheduling benchmark: {bench.name}")

        container = self.client.containers.run(
            image=bench.image,
            command=f"./run -a run -S {bench.library} -p {bench.name} -i native -n {bench.thread_num}",
            name=bench.name,
            cpuset_cpus=f"{cores[0]}-{cores[-1]}",
            detach=True,
            remove=False,
        )

        # Set the cores used to True
        for core in cores:
            self.cores_used[core] = True

        self.logger.job_start(
            bench.to_job(),
            initial_cores=cores,
            initial_threads=bench.thread_num
        )

        # Add job to the running queue
        job_manager.running.append(container)

    def _handle_update(self, bench: Benchmark, cores: List[int]):
        container = bench.container

        # Relinquish the previous cores
        self.relinquish_cores(container)

        # Update the cores used by the container
        container.update(cpuset_cpus=f"{cores[0]}-{cores[-1]}")

        # Set the new cores to True
        for core in cores:
            self.cores_used[core] = True

        self.logger.update_cores(Job._member_map_[container.name.upper()], cores)

    def _handle_pause(self, strategy: SchedulingStrategy, bench: Benchmark):
        job_manager = strategy.job_manager

        # Pause the container
        container = bench.container
        container.reload()

        if container.status != "exited":
            self.relinquish_cores(container)
            container.pause()

            # Update running and paused queues
            job_manager.paused.append(bench)
            job_manager.running.remove(bench)
            self.logger.job_pause(Job._member_map_[bench.name.upper()])        

    def _handle_unpause(self, strategy: SchedulingStrategy, bench: Benchmark):
        job_manager = strategy.job_manager

        # Unpause the container
        container = bench.container
        container.reload()

        if container.status != "exited":
            container.unpause()

            # Update paused and running queues
            job_manager.paused.remove(bench)
            job_manager.running.append(bench)
            self.logger.job_unpause(Job._member_map_[bench.name.upper()])

    def relinquish_cores(self, container: Container) -> None:
        core_start, core_end = container.attrs['HostConfig']['CpusetCpus'].split("-")
        for core in range(int(core_start), int(core_end) + 1):
            print(f"Relinquishing core {core}...")
            self.cores_used[core] = False

    def update_memcached(self, load: Load):
        cores = ["0"]
        if load == Load.HIGH:
            cores.append("1")
        
        subprocess.run(
            f"sudo taskset -a -cp {cores[0]}-{cores[-1]} {self.mc_process.pid}",
            shell=True
        )
        self.logger.update_cores(Job.MEMCACHED, cores)

    def flush_buffer(self, strategy: SchedulingStrategy):
        for command in strategy.command_buffer:
            match(command):
                case Run(command.bench, command.cores):
                    self._handle_run(strategy, command.bench, command.cores)
                case Pause(command.bench):
                    self._handle_pause(strategy, command.bench)
                case Unpause(command.bench):
                    self._handle_unpause(strategy, command.bench)
                case Update(command.bench, command.cores):
                    self._handle_update(command.bench, command.cores)


    def run_strategy(self, strategy: SchedulingStrategy):
        
        while True:
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
            if jobs_completed:
                strategy.on_job_complete(jobs_completed, state)
            elif state.load != self.current_load:
                self.update_memcached(state.load)  # Update memcached
                self.current_load = state.load
                strategy.on_state_update(state)

                # Flush the buffer and execute the commands accumulated
                self.flush_buffer(strategy)

            # # Relinquish cores from paused containers
            # for benchmark in self.job_manager.paused:
            #     self.relinquish_cores(benchmark.container)

def main():
    pass

if __name__ == "__main__":
    main()
