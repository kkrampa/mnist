import os
import psutil
import tensorflow as tf
import numpy as np
import time
import GPUtil

pid = os.getpid()
print('(Profiler) PID: ', pid)
py = psutil.Process(pid)


gpu_names = []
try:
    GPUs = GPUtil.getGPUs()
    for i, g in enumerate(GPUs):
        gpu_names.append('%d_%s'%(i,g.name))
except:
    pass

iostat_version='new'
if iostat_version=='new':
    disks = os.popen("iostat -dky |tail -n +4| awk '// { print $1}'").read().split()
else:
    disks = os.popen('iostat -d -o -n3 -K | head -1').read().split()

class netio(object):
    def __init__(self):
        self.time = time.time()
        psutil.net_io_counters.cache_clear()
        self.net_io = psutil.net_io_counters(nowrap=True)
    def get_stats(self):
        new_time = time.time()
        new_net_io = psutil.net_io_counters(nowrap=True)
        bytes_sent = float(new_net_io.bytes_sent-self.net_io.bytes_sent)/(new_time-self.time)
        bytes_recv = float(new_net_io.bytes_recv-self.net_io.bytes_recv)/(new_time-self.time)
        self.time = new_time
        self.net_io = new_net_io
        return bytes_sent, bytes_recv

global_net_io = netio()

def gpu_profiler():
    GPUs = GPUtil.getGPUs()
    values = []
    for gpu in GPUs:
        values.extend([float(gpu.load*100), float(gpu.memoryUtil*100)])
    return np.array(values)

def memory_profiler():
    return np.float(py.memory_info()[0]/2.**30)

def cpu_profiler():
    return np.float(py.cpu_percent())

def io_profiler():
    if iostat_version == 'new':
        values = [float(v) for v in os.popen("iostat -dky |tail -n +4| awk '// { print $2, $3, $4}'").read().split()]
    else:
        values = [float(v) for v in os.popen('iostat -d -o -n3 -K | tail -1').read().split()]
    return np.array(values)

def net_io_profiler():
    bs, br = global_net_io.get_stats()
    return np.float(bs/1000000), np.float(br/1000000)

def profiler_summaries(name='Profiler',
                    memory=True,
                    cpu_percent=True,
                    disk_io=True,
                    network_io=True,
                    GPU=True):
    summaries = []

    if memory:
        memory = tf.py_func(memory_profiler, [], tf.double)
        memory_summary = tf.summary.scalar('%s/MemoryGB'%name, memory)
        summaries.append(memory_summary)

    if cpu_percent:
        cpu_percent = tf.py_func(cpu_profiler, [], tf.double)
        cpu_percent_summary = tf.summary.scalar('%s/CPU/Percent'%name, cpu_percent)
        summaries.append(cpu_percent_summary)

    if disk_io:
        io_values = tf.py_func(io_profiler, [], tf.double)
        io_values.set_shape(3*len(disks))
        for i,disk in enumerate(disks):
            stps_summary = tf.summary.scalar('%s/IO/%s/SectorsTransferredPerSecond'%(name, disk), io_values[3*i])
            tps_summary = tf.summary.scalar('%s/IO/%s/TransfersPerSecond'%(name, disk), io_values[3*i+1])
            mpt_summary = tf.summary.scalar('%s/IO/%s/AvgMillisecondsPerTransaction'%(name, disk), io_values[3*i+2])
            summaries.extend([stps_summary, tps_summary, mpt_summary])

    if GPU and len(gpu_names)>0:
        gpu_values = tf.py_func(gpu_profiler, [], tf.double)
        gpu_values.set_shape(2*len(gpu_names))
        for i,gpu in enumerate(gpu_names):
            load_summary = tf.summary.scalar('%s/GPU/%s/Load'%(name, gpu), gpu_values[2*i])
            memory_summary = tf.summary.scalar('%s/GPU/%s/Memory'%(name, gpu), gpu_values[2*i+1])
            summaries.extend([load_summary, memory_summary])


    if network_io:
        bs, br = tf.py_func(net_io_profiler, [], [tf.double, tf.double])
        bs_summary = tf.summary.scalar('%s/NetIO/RecvMBPerSecond'%name, br)
        br_summary = tf.summary.scalar('%s/NetIO/SendMBPerSecond'%name, bs)
        summaries.extend([bs_summary, br_summary])

    return summaries
