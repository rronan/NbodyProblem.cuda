#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL NP_ARRAY_API
#include <numpy/arrayobject.h>
#include <time.h>

PyObject *init_space;

void setSpace(float *p, float *v, PyArrayObject *space_arr, int N)
{
    for (int i = 0; i < N; i++)
    {
        p[i] = *(double *)PyArray_GETPTR3(space_arr, i, 0, 0);
        p[i + N] = *(double *)PyArray_GETPTR3(space_arr, i, 1, 0);
        p[i + 2 * N] = *(double *)PyArray_GETPTR3(space_arr, i, 2, 0);
        v[i] = *(double *)PyArray_GETPTR3(space_arr, i, 0, 1);
        v[i + N] = *(double *)PyArray_GETPTR3(space_arr, i, 1, 1);
        v[i + 2 * N] = *(double *)PyArray_GETPTR3(space_arr, i, 2, 1);
    }
}

void writeSpace(float *p, int N)
{
    FILE *file = fopen("trajectories/result.data", "a");
    for (unsigned long i = 0; i < N; i++)
    {
        fprintf(file, "%f %f %f\n", p[i], p[i + N], p[i + N * 2]);
    }
    fwrite("\n", sizeof(char), 1, file);
    fclose(file);
}

__global__ void forwardGravitation(float *p, float *f, float G, float SOFTENING, int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = (int)((1 + sqrt(1 + 8 * (float)k)) / 2);
    int j = k - i * (i - 1) / 2;
    float dx = pow(p[i] - p[j], 2);
    float dy = pow(p[i + N] - p[j + N], 2);
    float dz = pow(p[i + 2 * N] - p[j + 2 * N], 2);
    float r = sqrt(dx + dy + dz + SOFTENING);
    float h = G / (r * r * r);
    f[k] = h * (p[j] - p[i]);
    f[k + N] = h * (p[j + N] - p[i + N]);
    f[k + 2 * N] = h * (p[j + 2 * N] - p[i + 2 * N]);
}

__global__ void sumAcceleration(float *v, float *f, float DT, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < N; j++)
    {
        if (j < i)
        {
            int k = i * (i - 1) / 2 + j;
            v[i] += DT * f[k];
            v[i + N] += DT * f[k + N];
            v[i + N * 2] += DT * f[k + N * 2];
        }
        else if (j > i)
        {
            int k = j * (j - 1) / 2 + i;
            v[i] -= DT * f[k];
            v[i + N] -= DT * f[k + N];
            v[i + N * 2] -= DT * f[k + N * 2];
        }
    }
}

__global__ void forwardPosition(float *p, float *v, float DT, int N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    p[n] += DT * v[n];
    p[n + N] += DT * v[n + N];
    p[n + 2 * N] += DT * v[n + 2 * N];
}

static PyObject *run(PyObject *Py_UNUSED(self), PyObject *args)
{
    printf("Start gravity_gpu!\n");
    fflush(stdout);
    int WRITE_INTERVAL, NSTEPS;
    float G, DT, SOFTENING, DAMPING;
    if (!PyArg_ParseTuple(args, "Oiffffi", &init_space, &NSTEPS, &G, &DT, &DAMPING, &SOFTENING, &WRITE_INTERVAL))
    {
        return NULL;
    }
    PyArrayObject *space_arr = (PyArrayObject *)PyArray_ContiguousFromObject(init_space, NPY_DOUBLE, 0, 0);
    int N = PyArray_DIMS(space_arr)[0];
    int N_lt = N * (N - 1) / 2;
    /* Init CPU arrays */
    float *p, *v;
    p = (float *)malloc(3 * sizeof(float) * N);
    v = (float *)malloc(3 * sizeof(float) * N);
    setSpace(p, v, space_arr, N);
    /* Init CUDA arrays */
    float *d_p, *d_v, *d_f;
    cudaMalloc(&d_p, 3 * sizeof(float) * N);
    cudaMalloc(&d_v, 3 * sizeof(float) * N);
    cudaMalloc(&d_f, 3 * sizeof(float) * N_lt);
    cudaMemcpy(d_p, p, 3 * sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, 3 * sizeof(float) * N, cudaMemcpyHostToDevice);
    FILE *file = fopen("trajectories/result.data", "w");
    fclose(file);
    time_t t0 = time(NULL);
    for (unsigned long i = 0; i < NSTEPS; i++)
    {
        forwardGravitation<<<1, N_lt>>>(d_p, d_f, G, SOFTENING, N);
        sumAcceleration<<<1, N>>>(d_v, d_f, DT, N);
        forwardPosition<<<1, N>>>(d_p, d_v, DT, N);
        if (i % WRITE_INTERVAL == 0)
        {
            printf("%ld\r", i);
            fflush(stdout);
            cudaMemcpy(p, d_p, 3 * sizeof(float) * N, cudaMemcpyDeviceToHost);
            writeSpace(p, N);
        }
    }
    printf("Total time: %f\n", (float)(time(NULL) - t0));
    free(p);
    free(v);
    cudaFree(d_p);
    cudaFree(d_v);
    cudaFree(d_f);
    return Py_None;
}

static PyMethodDef module_methods[] = {{"run", run, METH_VARARGS, NULL}, {0, 0}};

static struct PyModuleDef gravity_gpu = {PyModuleDef_HEAD_INIT, .m_name = "gravity_gpu", .m_methods = module_methods};

PyMODINIT_FUNC PyInit_gravity_gpu(void)
{
    Py_Initialize();
    import_array();
    return PyModule_Create(&gravity_gpu);
}
