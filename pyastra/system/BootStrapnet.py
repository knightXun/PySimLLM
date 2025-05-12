from mpi4py import MPI
import os

world_size = 0
local_rank = 0
rank2addr = {}


def initBootStrapNetRank(argc, argv):
    global rank2addr
    host_file = argv[1]
    try:
        with open(host_file, 'r') as file:
            rank = 0
            for line in file:
                rank2addr[rank] = line.strip()
                rank += 1
    except FileNotFoundError:
        print("Failed to open the file")


def BootStrapNet(argc, argv):
    global world_size, local_rank
    MPI.Init()
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    local_rank = comm.Get_rank()
    comm.Barrier()
    initBootStrapNetRank(argc, argv)
    