import argparse
import os
import json
import subprocess

# Utility to get name of instance
def get_instance_name(prefix: str):
    machine = subprocess.run(
        f"gcloud compute instances list --filter='name~^{prefix}' --format='value(name)'",
        text=True, shell=True, capture_output=True
    )
    return machine.stdout.strip().split("\n")[0]

# Update variables in bash script
def update_bash_script(
    file_path, 
    **kwargs
):
    with open(file_path, "r") as f:
        lines = f.readlines()

    with open(file_path, "w") as f:
        for line in lines:
            if line.startswith("TARG="):
                f.write(f"TARG={kwargs['targ']}\n")
            elif line.startswith("MEMCACHED_IP="):
                f.write(f"MEMCACHED_IP={kwargs['memcached']}\n")
            elif line.startswith("INTERNAL_AGENT_IP="):
                f.write(f"INTERNAL_AGENT_IP={kwargs['internalAgent']}\n")
            elif line.startswith("CORE_START="):
                f.write(f"CORE_START={kwargs['coreStart']}\n")
            elif line.startswith("CORE_END="):
                f.write(f"CORE_END={kwargs['coreEnd']}\n")
            elif line.startswith("SERVER="):
                f.write(f"SERVER={kwargs['server']}\n")
            else:
                f.write(line)

def compile_memcached_on_instance(instance: str, script_path: str="scripts/compile_memcache.bash"):
    subprocess.run(f"gcloud compute scp --ssh-key-file ~/.ssh/cloud-computing {script_path} ubuntu@{instance}:~/ --zone europe-west1-b", shell=True)
    subprocess.run(f"gcloud compute ssh --ssh-key-file ~/.ssh/cloud-computing ubuntu@{instance} --zone europe-west1-b --command='bash ~/$(basename {script_path})'", shell=True)
    subprocess.run("exit", shell=True)

def scp_start_memcached_script(
    instance: str,
    script_path: str,
    **kwargs
):
    
    if "client-agent" in instance:
        # Edit run_agent.bash script
        update_bash_script(script_path, targ=8)
    else:
        # Edit run_measure.bash script
        internalAgentIP = kwargs["internalAgent"]
        memcachedIP = kwargs["memcached"]

        update_bash_script(script_path, memcached=memcachedIP, internalAgent=internalAgentIP)
    
    subprocess.run(f"gcloud compute scp --ssh-key-file ~/.ssh/cloud-computing {script_path} ubuntu@{instance}:~/ --zone europe-west1-b", shell=True)

def get_internal_ip(target: str):
    result = subprocess.run(
        ["kubectl", "get", "node", target, "-o", "json"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"Error getting node {target}: {result.stderr}")
        return None

    node_info = json.loads(result.stdout)
    for addr in node_info["status"]["addresses"]:
        if addr["type"] == "InternalIP":
            return addr["address"]
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments to Launch Cluster on GCloud")
    parser.add_argument("--part", type=str, choices=["1", "2a", "2b", "3", "4"], help="Cluster to launch")
    args = parser.parse_args()

    # Add environment variables and create resources
    os.environ["KOPS_STATE_STORE"] = "gs://cca-eth-2025-group-73-jfleitas/"
    os.environ["PROJECT"] = "cca-eth-2025-group-73"
    subprocess.run(f"kops create -f part{args.part}.yaml", text=True, shell=True)

    # Deploy the cluster
    subprocess.run(f"kops update cluster --name part{args.part}.k8s.local --yes --admin", text=True, shell=True)

    # Validate the cluster
    subprocess.run("kops validate cluster --wait 10m", text=True, shell=True)

    # SSH into agent and measure servers
    agent = get_instance_name("client-agent")
    measure = get_instance_name("client-measure")
    server = get_instance_name("memcache-server")

    # SCP the memcached script and run it in the machines
    script_path = "scripts/setup/compile_memcache.bash"
    compile_memcached_on_instance(agent, script_path)
    compile_memcached_on_instance(measure, script_path)

    # Get internal IP addresses
    internal_agent_ip = get_internal_ip(agent)
    internal_measure_ip = get_internal_ip(measure)
    internal_server_ip = get_internal_ip(server)

    # Start memcached agent and client
    scp_start_memcached_script(agent, "scripts/setup/run_agent.bash", internalAgent=internal_agent_ip)
    scp_start_memcached_script(measure, "scripts/setup/run_measure_part4.bash", memcached=internal_server_ip, internalAgent=internal_agent_ip)

    # Launch memcached service
    script_path = "scripts/setup/run_memcached_part4.bash"
    subprocess.run(f"gcloud compute scp --ssh-key-file ~/.ssh/cloud-computing {script_path} ubuntu@{server}:~/ --zone europe-west1-b", shell=True)
    subprocess.run(f"gcloud compute ssh --ssh-key-file ~/.ssh/cloud-computing ubuntu@{server} --zone europe-west1-b --command='bash ~/$(basename {script_path})'", shell=True)

    # SCP scheduler and scheduler_logger files
    subprocess.run(f"gcloud compute scp --ssh-key-file ~/.ssh/cloud-computing scripts/scheduler.py ubuntu@{server}:~/ --zone europe-west1-b", shell=True)
    subprocess.run(f"gcloud compute scp --ssh-key-file ~/.ssh/cloud-computing scripts/scheduler_logger.py ubuntu@{server}:~/ --zone europe-west1-b", shell=True)
    update_bash_script("scripts/setup/send_logger.bash", server=internal_server_ip)

    # Output relevant cluster information to text file
    with open("scripts/setup/cluster.txt", "w") as f:
        f.write(f"AGENT NAME: {agent}\n")
        f.write(f"MEASURE NAME: {measure}\n\n")
        f.write(f"SERVER NAME: {server}\n\n")

        f.write(f"MEMCACHED IP ADDRESS: {internal_server_ip}\n")
        f.write(f"INTERNAL AGENT IP ADDRESS: {internal_agent_ip}\n")
        f.write(f"INTERNAL MEASURE SERVER IP ADDRESS: {internal_measure_ip}\n")
