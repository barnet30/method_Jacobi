#include <cmath>
#include <omp.h>
#include <iostream>

using namespace std;

float iteration_jacobi(float** alf, float* x, float* x0, float* beta, int n)
{
    float temp, max;
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        temp = 0;
        for (int j = 0; j < n; j++) {
            temp += alf[i][j] * x[j];
        }
        temp += beta[i];
        if (i == 0) {
            max = fabs(x[i] - temp);
        }
        else if (fabs(x[i] - temp) > max) {
            max = fabs(x[i] - temp);
        }
        x0[i] = temp;
    }
    return max;
}

int Jacobi(float** a, float* b, float* x, int n, float eps) {
    float** alf, * beta, * x0, norm;
    int numbIter;
    alf = new float* [n];
    for (int i = 0; i < n; i++)
        alf[i] = new float[n];
    beta = new float[n];
    x0 = new float[n];
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            if (i == j)
                alf[i][j] = 0;
            else
                alf[i][j] = -a[i][j] / a[i][i];
        beta[i] = b[i] / a[i][i];
    }
    for (int i = 0; i < n; i++)
        x0[i] = beta[i];
    numbIter = 0;
    norm = 5 * eps;
    while (norm > eps) {
        for (int i = 0; i < n; i++)
            x[i] = x0[i];
        norm = iteration_jacobi(alf, x, x0, beta, n);
        numbIter++;
    }
    for (int i = 0; i < n; i++)
        delete[] alf[i];
    delete[] alf;
    delete[] beta;
    delete[] x0;
    return numbIter;
}

int main() {
    setlocale(LC_ALL, "ru");
    int cores[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    int sizes[] = { 500, 1000, 2500, 5000, 7500, 10000, 15000 };
    float** a;
    float* b;
    float* x;
    float eps;
    eps = 1e-6;
    int numbIteration;

    for (auto size : sizes) {
        cout << "время для N = " << size << ":" << endl;
        for (auto core : cores) {
            omp_set_num_threads(core);
            a = new float* [size];
            for (int i = 0; i < size; i++)
                a[i] = new float[size];
            b = new float[size];
            x = new float[size];
            for (int i = 0; i < size; a[i][i] = 1, i++)
                for (int j = 0; j < size; j++)
                    if (i != j)
                        a[i][j] = 0.1 / (i + j);
            for (int i = 0; i < size; i++)
                b[i] = sin(i);

            float t1 = omp_get_wtime();
            numbIteration = Jacobi(a, b, x, size, eps);
            float t2 = omp_get_wtime();
            cout << t2 - t1 << endl;
            cout << "потоков = " << core << endl;
            cout <<"Time = "<< t2 - t1 << endl;
            cout << "Искомый вектор X:" << endl;
            cout << x[0] << "\t" << x[size / 2] << "\t" << x[size - 1] << endl;
            cout.flush();
            for (int i = 0; i < size; i++)
                delete[]a[i];
            delete[] a;
            delete[] b;
            delete[] x;
        }
        cout << endl;
    }
    return 0;
}