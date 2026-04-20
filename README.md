# ROCM
This is just some playground code I've been trying out to learn the rocm basics. Basic memcpy logic, kernel logic
and trying to understand some of the LLM libraries. I would like to investigate some of the algorithms used and 
(as I understand, linear algebra) math involved with such workflows.

Note that I'm running a <b>Radeon RX 7900 XT</b> so some of the code may be tailored to that specific architecture as I try out some optimization.



## Execution

```shell
# compile
make

# run the code
make run

# remove build files
make clean
```

## Links

[ROCM Documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/initialization.html#querying-and-setting-gpus)

[ROCM Demo Repo](https://github.com/ROCm/rocm-examples#)

[ROCM Intro Videos](https://www.youtube.com/playlist?list=PLx15eYqzJifehAxhWRD6T35GZwAqM9IK4)