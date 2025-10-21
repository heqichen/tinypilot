# export TVM_HOME=/media/storage/workspace/tvm
# export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
# export LD_LIBRARY_PATH=$TVM_HOME/lib:$LD_LIBRARY_PATH


# device = mali
import tvm
import numpy as np
from tvm import te

def test_mali():
    target = "opencl --device=mali"
    dev = tvm.device(target, 0)
    
    # Construct model
    
    InputA = te.placeholder(shape=(1,),dtype="float32", name="InputA")
    InputB = te.placeholder(shape=(1,),dtype="float32", name="InputB")
    Output = te.compute(shape=(1,), fcompute=lambda i: InputA[i] + InputB[i], name="Output")
    func = te.create_prim_func([InputA, InputB, Output])
    
    sch = tvm.tir.Schedule(func)
    (i, ) = sch.get_loops(sch.get_block("Output"))
    sch.bind(i, "threadIdx.x")
    exe = tvm.tir.build(sch.mod, target=target)
    
    a = tvm.nd.array(np.array([1], dtype="float32"), dev)
    b = tvm.nd.array(np.array([200], dtype="float32"), dev)
    c = tvm.nd.empty((1,), dtype="float32", device=dev)
    
    exe(a, b, c)
    print(c)
    
    exe.export_library("output.so")
    
    loaded_lib = tvm.runtime.load_module("output.so")
    ia = tvm.nd.array(np.array([567], dtype="float32"), dev)
    ib = tvm.nd.array(np.array([246], dtype="float32"), dev)
    ic = tvm.nd.empty((1,), dtype="float32", device=dev)
    
    loaded_lib(ia, ib, ic)
    print(ic)
    
    
test_mali()