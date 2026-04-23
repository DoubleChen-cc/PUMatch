#!/bin/bash

# 循环运行测试从 1.g 到 24.g
for i in {3..23}; do
    echo "Running test with pattern: $i.g"
    mpirun -np 2 -H node3:1,node1:1 --bind-to none  --mca pml ob1   --mca btl self,tcp   --mca btl_tcp_if_include 192.168.254.0/24 ./GAMMA-main/sm_mpi data/com-friendster/snap.txt pattern1/$i.g  3 full 
    echo "Completed pattern $i.g"
    echo "-------------------"
done

echo "All tests completed!"

for i in {3..23}; do
    echo "Running test with pattern: $i.g"
    mpirun -np 3 -H node3:1,node1:1,node0:1 --bind-to none  --mca pml ob1   --mca btl self,tcp   --mca btl_tcp_if_include 192.168.254.0/24 ./GAMMA-main/sm_mpi data/com-friendster/snap.txt pattern1/$i.g  3 full 
    echo "Completed pattern $i.g"
    echo "-------------------"
done