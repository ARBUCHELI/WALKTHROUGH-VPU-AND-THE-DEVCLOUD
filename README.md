# WALKTHROUGH-VPU-AND-THE-DEVCLOUD

## This Content was Created by Intel Edge AI for IoT Developers UDACITY Nanodegree.

This notebook is a demonstration showing how to request an edge node with an Intel i5 CPU and load a model on the VPU (Intel® Neural Compute Stick 2) using Udacity's workspace 
integration with Intel's DevCloud.

Below are the six steps we'll walk through in this notebook:

1. Creating a Python script to load the model
2. Creating a job submission script
3. Submitting a job using the <code>qsub</code> command
4. Checking the job status using the <code>liveQStat</code> function
5. Retrieving the output files using the <code>getResults</code> function
6. Viewing the resulting output

<strong>IMPORTANT: Set up paths so we can run Dev Cloud utilities</strong>

You must run this every time you enter a Workspace session.

<pre><code>
%env PATH=/opt/conda/bin:/opt/spark-2.4.3-bin-hadoop2.7/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/intel_devcloud_support
import os
import sys
sys.path.insert(0, os.path.abspath('/opt/intel_devcloud_support'))
sys.path.insert(0, os.path.abspath('/opt/intel'))
</code></pre>

# The Model
We will be using the <code>vehicle-license-plate-detection-barrier-0106</code> model for this exercise. Remember that to run a model on the VPU (Intel® NCS2), we need to use 
<code>FP16</code> as the model precision.

The model has already been downloaded for you in the <code>/data/models/intel</code> directory on Intel's DevCloud. We will be using the following filepath during the job 
submission in <code>Step 3</code>:

<code>/data/models/intel/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106</code>

# Step 1: Creating a Python Script
The first step is to create a Python script that you can use to load the model and perform an inference. I have used the <code>%%writefile</code> magic command to create a 
Python file called <code>load_model_to_device.py</code>. This will create a new Python file in the working directory.

<pre><code>
%%writefile load_model_to_device.py

import time
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IECore
import argparse

def main(args):
    model=args.model_path
    model_weights=model+'.bin'
    model_structure=model+'.xml'
    
    start=time.time()
    model=IENetwork(model_structure, model_weights)

    core = IECore()
    net = core.load_network(network=model, device_name=args.device, num_requests=1)
    print(f"Time taken to load model = {time.time()-start} seconds")


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--device', default=None)
    
    args=parser.parse_args() 
    main(args)
</code></pre>

# Step 2: Creating a Job Submission Script
To submit a job to the DevCloud, we need to create a shell script. Similar to the Python script above, I have used the <code>%%writefile</code> magic command to create a shell 
script called <code>load_ncs_model_job.sh</code>.

This script does a few things.

1. Writes stdout and stderr to their respective .log files
2. Creates the <code>/output</code> directory
3. Creates <code>DEVICE</code> and <code>MODELPATH</code> variables and assigns their value as the first and second argument passed to the shell script
4. Calls the Python script using the <code>MODELPATH</code> and <code>DEVICE</code> variable values as the command line argument
5. Changes to the <code>/output</code> directory
6. Compresses the stdout.log and stderr.log files to <cod>output.tgz</code>

<pre><code>
%%writefile load_ncs_model_job.sh

exec 1>/output/stdout.log 2>/output/stderr.log

mkdir -p /output

DEVICE=$1
MODELPATH=$2

# Run the load model python script
python3 load_model_to_device.py  --model_path ${MODELPATH} --device ${DEVICE}

cd /output

tar zcvf output.tgz stdout.log stderr.log
</code></pre>

# Step 3: Submitting a Job to Intel's DevCloud 
The code below will submit a job to an <strong>IEI Tank-870</strong> edge node with an Intel® i5 processor and Intel® NCS2. We will load the model on the VPU.

<strong>Note</strong>: We'll pass in a device type argument of <code>MYRIAD</code> to load our model on the VPU (Intel® NCS2). As a reminder, when running a model on a VPU, the
model precision we'll need <code>FP16</code>.

The <code>!qsub</code> command takes a few command line arguments:

1. The first argument is the shell script filename - <code>load_ncs_model_job.sh</code>. This should always be the first argument.
2. The <code>-d</code> flag designates the directory where we want to run our job. We'll be running it in the current directory as denoted by <code>.</code>.
3. The <code>-l</code> flag designates the node and quantity we want to request. The default quantity is 1, so the 1 after <code>nodes</code> is optional.
4. The <code>-F</code> flag let's us pass in a string with all command line arguments we want to pass to our Python script.

<strong>Note</strong>: There is an optional flag, <code>-N</code>, you may see in a few exercises. This is an argument that only works on Intel's DevCloud that allows you to 
name your job submission. This argument doesn't work in Udacity's workspace integration with Intel's DevCloud.

In the cell below, we assign the returned value of the <code>!qsub</code> command to a variable <code>job_id_core</code>. This value is an array with a single string.

Once the cell is run, this queues up a job on Intel's DevCloud and prints out the first value of this array below the cell, which is the job id.

<pre><code>
job_id_core = !qsub load_ncs_model_job.sh -d . -l nodes=1:tank-870:i5-6500te:intel-ncs2 -F "MYRIAD /data/models/intel/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106"
print(job_id_core[0])
</code></pre>

# Step 4: Running liveQStat
Running the <code>liveQStat</code> function, we can see the live status of our job. Running the this function will lock the cell and poll the job status 10 times. The cell is 
locked until this finishes polling 10 times or you can interrupt the kernel to stop it by pressing the stop button at the top:

* <code>Q</code> status means our job is currently awaiting an available node
* <code>R</code> status means our job is currently running on the requested node

<strong>Note</strong>: In the demonstration, it is pointed out that <code>W</code> status means your job is done. This is no longer accurate. Once a job has finished running, 
it will no longer show in the list when running the <code>liveQStat</code> function.

<pre><code>
import liveQStat
liveQStat.liveQStat()
</code></pre>

# Step 5: Retrieving Output Files 
In this step, we'll be using the <code>getResults</code> function to retrieve our job's results. This function takes a few arguments.

1. <code>job id</code> - This value is stored in the <code>job_id_core</code> variable we created during <strong>Step 3</strong>. Remember that this value is an array with a 
single string, so we access the string value using <code>job_id_core[0]</code>.
2. <code>filename</code> - This value should match the filename of the compressed file we have in our <code>load_ncs_model_job.sh</code> shell script. In this example, filename
shoud be set to <code>output.tgz</code>.
3. <code>blocking</code> - This is an optional argument and is set to False by default. If this is set to True, the cell is locked while waiting for the results to come back. 
There is a status indicator showing the cell is waiting on results.

<strong>Note</strong>: The <code>getResults</code> function is unique to Udacity's workspace integration with Intel's DevCloud. When working on Intel's DevCloud environment, 
your job's results are automatically retrieved and placed in your working directory.

<pre><code>
import get_results

get_results.getResults(job_id_core[0], filename="output.tgz", blocking=True)
</code></pre>

# Step 6: Viewing the Outputs
In this step, we unpack the compressed file using <code>!tar zxf</code> and read the contents of the log files by using the <code>!cat</code> command.

<code>stdout.log</code> should contain the printout of the print statement in our Python script.

<code>!tar zxf output.</code>
<code>!cat stdout.</code>
<code>!cat stderr.log</code>

# Adaptation as a Repository: Andrés R. Bucheli.












