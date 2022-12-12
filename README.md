# SC & PDP
***Суперкомпьютеры и параллельная обработка данных, 4 курс ВМК***

## Условие 
Необходимо реализовать параллельную версию **алгоритма Гаусса** решения СЛАУ при помощи технологий **OpenMP** и **MPI**.
- Запустить программы на машине **IBM Polus**.
- Исследовать масштабируемость полученной параллельной программы:
    - построить графики зависимости времени выполнения программы от числа потоков/процессов и размерности входной матрицы.
- Для каждого набора входных данных найти количество потоков/процессов, при котором время выполнения задачи перестаёт уменьшаться.
- Определить основные причины недостаточной масштабируемости программы при максимальном числе используемых потоков/процессов.
- Сравнить эффективность распараллеливания программы средствами OpenMP и MPI.

## OpenMP
### Компиляция программы:
**1. Компилятор gcc**
```
gcc parallel_gauss_omp.c -o gauss -fopenmp
```
**2. Компилятор xlc**
```
xlc parallel_gauss_omp.c -o gauss -fopenmp
```
### Запуск программы:
**1. Написание OMP_runScript.lsf файла**
```
#BSUB -J "OMP_gaussTask"
#BSUB -o "OMP_gaussTask%J.out"
#BSUB -e "OMP_gaussTask%J.err"
#BSUB -R "affinity[core(M)]"
OMP_NUM_THREADS=N
/polusfs/lsf/openmp/launchOpenMP.py ./gauss
```
Где ***M*** - это количество ядер, а ***N*** - количество используемых нитей.
**2. Постановка задачи в очередь**
```
bsub < OMP_runScript.lsf
```

## MPI
Для создания и запуска программ, написанных с использованием MPI стандарта, необходимо загрузить модуль **SpectrumMPI**. Для этого необходимо выполнить следующую команду:
```
module load SpectrumMPI/10.1.0
```
### Компиляция программы:
**1. Компилятор mpicc**
```
mpicc parallel_gauss_mpi.c -o gauss
```
**2. Компилятор mpixlc**
```
mpixlc parallel_gauss_mpi.c -o gauss
```
### Запуск программы:
**1. Написание MPI _runScript.lsf файла**
```
#BSUB -n N
#BSUB -q normal
#BSUB -W time
#BSUB -J "MPI_gaussTask"
#BSUB -o "MPI_gaussTask.%J.out"
#BSUB -e "MPI_gaussTask.%J.err"
mpiexec ./gauss
```
Где ***N*** - это количество ядер(процессов), а ***time*** - время выполнения задачи (тайм-аут).
**2. Постановка задачи в очередь**
```
bsub < MPI_runScript.lsf
```

## Результаты
Отчеты с результатми доступны в папке [Repots](https://github.com/tsirleo/SC_PDP/tree/main/Reports).
Файлы с реализованным распараллеленным алгоритмом доступны в папке [Parallel code](https://github.com/tsirleo/SC_PDP/tree/main/Parallel_code)

