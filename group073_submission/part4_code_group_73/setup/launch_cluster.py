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
            elif line.startswith("INTERNAL_AGENT_A_IP="):
                f.write(f"INTERNAL_AGENT_A_IP={kwargs['internalAgentA']}\n")
            elif line.startswith("INTERNAL_AGENT_B_IP="):
                f.write(f"INTERNAL_AGENT_A_IP={kwargs['internalAgentB']}\n")
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
    
    if "client-agent-a" in instance:
        # Edit run_agent.bash script
        update_bash_script(script_path, targ=2)
    elif "client-agent-b" in instance:
        # Edit run_agent.bash script
        update_bash_script(script_path, targ=4)
    else:
        # Edit run_measure.bash script
        internalAgentAIP = kwargs["internalAgentA"]
        internalAgentBIP = kwargs["internalAgentB"]
        memcachedIP = kwargs["memcached"]

        update_bash_script(script_path, memcached=memcachedIP, internalAgentA=internalAgentAIP, internalAgentB=internalAgentBIP)
    
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

def launch_memcached(part: int, threads=1, cores= 1):    
    subprocess.run("kubectl create -f memcache-t1-cpuset.yaml", shell=True, text=True)
    subprocess.run("kubectl expose pod some-memcached --name some-memcached-11211 --type LoadBalancer --port 11211 --protocol TCP", shell=True, text=True)
    subprocess.run("sleep 60", shell=True, text=True)

    # Get the pod IP in JSON format and parse it
    result = subprocess.run(
        "kubectl get pod some-memcached -o json",
        shell=True,
        capture_output=True,
        text=True
    )

    # Parse the IP from the JSON output
    pod_info = json.loads(result.stdout)
    pod_ip = pod_info["status"].get("podIP")

    return pod_ip

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
    agentA = get_instance_name("client-agent-a")
    agentB = get_instance_name("client-agent-b")
    measure = get_instance_name("client-measure")

    # SCP the memcached script and run it in the machines
    script_path = "scripts/setup/compile_memcache.bash"
    compile_memcached_on_instance(agentA, script_path)
    compile_memcached_on_instance(agentB, script_path)
    compile_memcached_on_instance(measure, script_path)

    # Launch memcached service
    memcached_ip = launch_memcached()

    # Get internal IP addresses
    internal_agentA_ip = get_internal_ip(agentA)
    internal_agentB_ip = get_internal_ip(agentB)
    internal_measure_ip = get_internal_ip(measure)

    scp_start_memcached_script(agentA, "scripts/setup/run_agent.bash")
    scp_start_memcached_script(agentB, "scripts/setup/run_agent.bash")
    scp_start_memcached_script(measure, "scripts/setup/run_measure.bash", memcached=memcached_ip, internalAgentA=internal_agentA_ip, internalAgentB=internal_agentB_ip)

    # Output relevant cluster information to text file
    with open("scripts/setup/cluster.txt", "w") as f:
        f.write(f"AGENT A NAME: {agentA}\n")
        f.write(f"AGENT A NAME: {agentB}\n")
        f.write(f"AGENT A NAME: {measure}\n\n")

        f.write(f"MEMCACHED IP ADDRESS: {memcached_ip}\n")
        f.write(f"INTERNAL AGENT A IP ADDRESS: {internal_agentA_ip}\n")
        f.write(f"INTERNAL AGENT A IP ADDRESS: {internal_agentB_ip}\n")
        f.write(f"INTERNAL MEASURE SERVER IP ADDRESS: {internal_measure_ip}\n")
