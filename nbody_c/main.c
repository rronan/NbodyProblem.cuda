#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL NP_ARRAY_API
#include <numpy/arrayobject.h>

unsigned long NSTEPS, WRITE_INTERVAL, NBODIES;
double G, DT, DAMPING, SOFTENING;

unsigned long dv_index(unsigned long x, unsigned long y, unsigned long z)
{
    unsigned long res = x * NBODIES * 3 + y * 3 + z;
    return res;
}

struct Body
{
    double *x;
    double *y;
    double *z;
    double *vx;
    double *vy;
    double *vz;
};

PyObject *init_space;

void setSpace(struct Body *bodies[], PyArrayObject *space_arr)
{
    for (unsigned long i = 0; i < NBODIES; i++)
    {
        bodies[i] = (struct Body *)malloc(sizeof(struct Body));
        bodies[i]->x = (double *)PyArray_GETPTR3(space_arr, i, 0, 0);
        bodies[i]->y = (double *)PyArray_GETPTR3(space_arr, i, 1, 0);
        bodies[i]->z = (double *)PyArray_GETPTR3(space_arr, i, 2, 0);
        bodies[i]->vx = (double *)PyArray_GETPTR3(space_arr, i, 0, 1);
        bodies[i]->vy = (double *)PyArray_GETPTR3(space_arr, i, 1, 1);
        bodies[i]->vz = (double *)PyArray_GETPTR3(space_arr, i, 2, 1);
    }
}

void writeSpace(struct Body *bodies[])
{
    FILE *f = fopen("result.data", "a");
    for (unsigned long i = 0; i < NBODIES; i++)
    {
        fprintf(f, "%f %f %f\n", *bodies[i]->x, *bodies[i]->y, *bodies[i]->z);
    }
    fwrite("\n", sizeof(char), 1, f);
    fclose(f);
}

void forwardGravitation(unsigned long i, unsigned long j, struct Body *bodies[], double *dv)
{
    struct Body *a = bodies[i];
    struct Body *b = bodies[j];
    double px = pow(*a->x - *b->x, 2);
    double py = pow(*a->y - *b->y, 2);
    double pz = pow(*a->z - *b->z, 2);
    double r = pow(px + py + pz + SOFTENING, .5);
    // pow (3 * .5)
    double h = DT * G / (r * r * r);
    dv[dv_index(i, j, 0)] = h * (*b->x - *a->x);
    dv[dv_index(i, j, 1)] = h * (*b->y - *a->y);
    dv[dv_index(i, j, 2)] = h * (*b->z - *a->z);
}

void sumAcceleration(struct Body *bodies[], double *dv)
{
#pragma omp parallel for
    for (unsigned long i = 0; i < NBODIES; i++)
    {
        for (unsigned long j = 0; j < i; j++)
        {
            *bodies[i]->vx += dv[dv_index(i, j, 0)];
            *bodies[i]->vy += dv[dv_index(i, j, 1)];
            *bodies[i]->vz += dv[dv_index(i, j, 2)];
            *bodies[j]->vx -= dv[dv_index(i, j, 0)];
            *bodies[j]->vy -= dv[dv_index(i, j, 1)];
            *bodies[j]->vz -= dv[dv_index(i, j, 2)];
        }
    }
}

void forwardSquare(struct Body *bodies[], double *dv)
{
#pragma omp parallel for
    for (unsigned long i = 0; i < NBODIES / 2; i++)
    {
        for (unsigned long j = 0; j < NBODIES / 2; j++)
        {
            unsigned long ii = NBODIES / 2 + i;
            unsigned long jj = NBODIES / 2 + j;
            forwardGravitation(ii, j, bodies, dv);
            if (i > j)
            {
                forwardGravitation(i, j, bodies, dv);
            }
            else
            {
                forwardGravitation(jj, ii, bodies, dv);
            }
        }
    }
    sumAcceleration(bodies, dv);
}

void forwardTriangle(struct Body *bodies[], double *dv)
{
    for (unsigned long i = 0; i < NBODIES; i++)
    {
        for (unsigned long j = 0; j < i; j++)
        {
            forwardGravitation(i, j, bodies, dv);
        }
    }
    sumAcceleration(bodies, dv);
}

void forwardPosition(struct Body *bodies[])
{
#pragma omp parallel for
    for (unsigned long i = 0; i < NBODIES; i++)
    {
        *bodies[i]->x += DT * (*bodies[i]->vx);
        *bodies[i]->y += DT * (*bodies[i]->vy);
        *bodies[i]->z += DT * (*bodies[i]->vz);
    };
}

static PyObject *run(PyObject *Py_UNUSED(self), PyObject *args)
{
    printf("Start gravity_cpu!\n");
    fflush(stdout);
    if (!PyArg_ParseTuple(args, "Olddddl", &init_space, &NSTEPS, &G, &DT, &DAMPING, &SOFTENING, &WRITE_INTERVAL))
    {
        return NULL;
    }
    PyArrayObject *space_arr = (PyArrayObject *)PyArray_ContiguousFromObject(init_space, NPY_DOUBLE, 0, 0);
    NBODIES = PyArray_DIMS(space_arr)[0];
    struct Body *bodies[NBODIES];
    double *dv = (double *)malloc(sizeof(double) * NBODIES * NBODIES * 3);
    setSpace(bodies, space_arr);
    FILE *f = fopen("result.data", "w");
    fclose(f);
    double total_time = 0;
    time_t t0 = time(NULL);
    for (unsigned long i = 0; i < NSTEPS; i++)
    {
        clock_t t = clock();
        forwardSquare(bodies, dv);
        forwardPosition(bodies);
        total_time += (double)(clock() - t);
        if (i % WRITE_INTERVAL == 0)
        {
            printf("%ld\r", i);
            fflush(stdout);
            writeSpace(bodies);
        }
    }
    time_t t1 = time(NULL);
    printf("CPU time: %f seconds\n", total_time / CLOCKS_PER_SEC);
    printf("Total time: %f seconds\n", (float)(time(NULL) - t0));
    free(dv);
    return Py_None;
}

static PyMethodDef module_methods[] = {{"run", run, METH_VARARGS, NULL}, {0, 0}};

static struct PyModuleDef gravity_cpu = {PyModuleDef_HEAD_INIT, .m_name = "gravity_cpu", .m_methods = module_methods};

PyMODINIT_FUNC PyInit_gravity_cpu(void)
{
    Py_Initialize();
    import_array();
    return PyModule_Create(&gravity_cpu);
}
