from pathlib import Path
import subprocess
import re
import tqdm
import yaml

def extract_real_time(text):
    match = re.search(r"real\s+(\d+)m([\d.]+)s", text)

    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        total_seconds = minutes * 60 + seconds
        return total_seconds

    return None

def get_logs_cmd(job_name: str):
    return (
        f"kubectl logs $(kubectl get pods --selector=job-name={job_name}"
        + " --output=jsonpath='{.items[*].metadata.name}')"
    )

def wait_for_job_completion(job_name: str, timeout_seconds=3600):
    """Wait for the Kubernetes job to complete."""
    wait_cmd = f"kubectl wait --for=condition=complete job/{job_name} --timeout={timeout_seconds}s"
    result = subprocess.run(wait_cmd, shell=True, text=False)
    return result.returncode == 0


def run_benchmark():
    part2b_dir = Path("parsec-benchmarks/part2b")
    prev = 2
    
    for num_threads in [4, 8]:
        file_name = f"./logs/part2b/parallel_log_{num_threads}.txt"
        for program_file in (t := tqdm.tqdm(list(part2b_dir.iterdir()))):
            job_name, _ = program_file.name.split(".")
            t.set_description(f"running {job_name}")
            with open(program_file, 'r') as file:
                data = yaml.safe_load(file)
            args = data['spec']['template']['spec']['containers'][0]['args']
            command_string = args[1]
            updated_command = command_string.replace(f'-n {prev}', f'-n {num_threads}')
            data['spec']['template']['spec']['containers'][0]['args'][1] = updated_command
            
            with open(program_file, 'w') as file:
                yaml.dump(data, file, default_flow_style=False)

            start_job_cmd = f"kubectl create -f {program_file}"
            _ = subprocess.run(start_job_cmd, text=True, shell=True)

            wait_for_job_completion(job_name)

            logs = subprocess.run(
                get_logs_cmd(job_name),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                shell=True,
            )
            real_time = extract_real_time(logs.stdout)
            with open(file_name, "a")  as f:
                f.write(f"{job_name}: {real_time}\n")

        prev = num_threads

        # Cleanup
        subprocess.run("kubectl delete jobs --all", shell=True)
        subprocess.run("kubectl delete pods --all", shell=True)

if __name__ == "__main__":
    run_benchmark()
