# Known issues

## HIP
- Compatibility issue between HIP and gcc 8. You may encounter the following error:
```
error: reference to __host__ function 'operator new' in __host__ __device__ function
```
gcc 7, 9, and later do not have this issue.
