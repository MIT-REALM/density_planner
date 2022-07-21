import os, time
import numpy as np
from multiprocessing.pool import ThreadPool as PP
import argparse

#np.show_config()

os.environ['MKL_DEBUG_CPU_TYPE']="5"

def worker_func(x):
    nt = args.nt
    dt = 0.001
    idx, s = x
    s_list = [s]
    # randomly data generation
    for ti in range(nt):
        grad = s ** 2 if ti % 2 == 0 else - s **2
        s = s + grad * dt
        s_list.append(s)
    s_list = np.stack(s_list, axis=1)
    return idx, s_list


def main():
    num_batch = args.batchsize
    pool = PP(args.workers)
    data = np.random.rand(num_batch, 10000)
    data_input = [[i, data[i]] for i in range(num_batch)]
    tt1 = time.time()
    data_out = pool.map(worker_func, data_input)
    out_list=[None for i in range(num_batch)]
    for i in range(num_batch):
        out_list[data_out[i][0]]=data_out[i][1]
    tt2 = time.time()
    out_list = np.stack(out_list, axis=0)
    print("Inner func took %.4f seconds. shape %s" % (tt2-tt1, out_list.shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test threading to improve data preprocessing")
    parser.add_argument("--nt", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    t1 = time.time()
    main()
    t2 = time.time()
    print("Finished in %.4f seconds" % (t2 - t1))