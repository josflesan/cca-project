import dataclasses
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import subprocess
import yaml
import argparse
from typing import List

from utils.part3_utils import get_time

@dataclasses.dataclass
class Job:
    name: str
    priority: int
    node: str
    isolated: bool = False
    thread_num: int = 1
    coresReq: int | None = None
    memoryReq: int | None = None
    coresLim: int | None = None
    memoryLim: int | None = None

    def __post_init__(self):
        self.path_name = f"parsec-benchmarks/part3/parsec-{self.name}.yaml"

jobs = [
    Job(
        name="dedup",
        priority=1,
        isolated=False,
        thread_num=2,
        node="node-a-2core",
        # coresLim=1,
        # memoryLim=8,
    ),
    Job(
        name="blackscholes",
        priority=1,
        isolated=True,
        thread_num=2,
        node="node-b-2core",
        # coresLim=1,
        # memoryLim=1,
    ),
    Job(
        name="ferret",
        priority=2,
        isolated=True,
        thread_num=4,
        node="node-d-4core",
        # coresReq=3,
        # memoryReq=14,
    ),
    Job(
        name="freqmine",
        priority=1,
        isolated=True,
        thread_num=4,
        node="node-c-4core",
        # coresReq=4,
        # memoryReq=8,
        # coresLim=4,
        # memoryLim=8,
    ),
    Job(
        name="radix",
        priority=1,
        isolated=True,
        thread_num=4,
        node="node-d-4core",
        # coresLim=1,
        # memoryLim=2,
    ),
    Job(
        name="vips",
        priority=2,
        isolated=True,
        thread_num=2,
        node="node-b-2core",
        # coresReq=2,
        # memoryReq=2,
    ),
    Job(
        name="canneal",
        priority=2,
        isolated=False,
        thread_num=2,
        node="node-a-2core",
        # coresReq=2,
        # memoryReq=8,
        # coresLim=2,
        # memoryLim=8,
    ),
]

def wait_for_job_completion(job_name: str, timeout_seconds: int = 3600):
    """Wait for the Kubernetes job to complete."""
    wait_cmd = f"kubectl wait --for=condition=complete job/parsec-{job_name} --timeout={timeout_seconds}s"
    result = subprocess.run(wait_cmd, shell=True, text=False)
    return result.returncode == 0


def schedule_jobs(jobs: List[Job]):
    jobs = sorted(jobs, key=lambda j: j.priority)
    for job in jobs:
        start_job_cmd = f"kubectl create -f {job.path_name}"
        print(f"starting {job.name}...")
        _ = subprocess.run(start_job_cmd, text=False, shell=True)
        if job.isolated:
            print(f"waiting for {job.name}...")
            wait_for_job_completion(job.name)
            print(f"done waiting for {job.name}...")
    # not sure about this
    for job in jobs:
        if not job.isolated:
            wait_for_job_completion(job.name)


def update_yaml(job):
    with open(job.path_name, "r") as file:
        data = yaml.safe_load(file)
        spec = data["spec"]["template"]["spec"]

        # Change resources
        if job.coresReq:
            spec["containers"][0]["resources"]["requests"]["cpu"] = str(job.coresReq)
        if job.memoryReq:
            spec["containers"][0]["resources"]["requests"]["memory"] = f"{job.memoryReq}Gi"
        if job.coresLim:
            spec["containers"][0]["resources"]["limits"]["cpu"] = str(job.coresLim)
        if job.memoryLim:
            spec["containers"][0]["resources"]["limits"]["memory"] = f"{job.memoryLim}Gi"

        # change node
        spec["nodeSelector"]["cca-project-nodetype"] = job.node

        # change thread count
        library = "parsec" if job.name != "radix" else "splash2x"
        run_cmd = f"./run -a run -S {library} -p {job.name} -i native -n {job.thread_num}"
        spec["containers"][0]["args"][1] = run_cmd

    # write back
    with open(job.path_name, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def main(run: int):
    nodes_to_jobs = defaultdict(list)
    for job in jobs:
        update_yaml(job)
        nodes_to_jobs[job.node].append(job)
    
    with ProcessPoolExecutor(4) as exec:
        exec.map(schedule_jobs, nodes_to_jobs.values())

    # Run commands to get final time:
    # $ kubectl get pods -o json > results.json
    # $ python3 get_time.py results.json
    subprocess.run(f"kubectl get pods -o json > logs/pods_{run}.json", shell=True)
    get_time(f"logs/pods_{run}.json")

    # Cleanup
    subprocess.run("kubectl delete jobs --all", shell=True)
    subprocess.run("kubectl delete pods --all", shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, choices=[1, 2, 3], help="Pods run")
    args = parser.parse_args()

    main(args.run)
