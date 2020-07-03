#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define MASTER 0

float *create1DArray(int n)
{
    float *T = (float *)malloc(n * sizeof(float));
    return T;
}

int *create1DIntArray(int n)
{
    int *T = (int *)malloc(n * sizeof(int));
    return T;
}

void fillArray(float *T, int n)
{
    int i;
    for (i = 0; i < n; i++)
        T[i] = 1.0;
}

void fillArrayWithOne(int *T, int n)
{
    int i;
    for (i = 0; i < n; i++)
        T[i] = 1;
}

void fillArrayWithZero(float *T, int n)
{
    int i;
    for (i = 0; i < n; i++)
        T[i] = 0;
}

void printArray(float *T, int n)
{
    int i;
    for (i = 0; i < n; i++)
        printf("%.2f ", T[i]);
    puts("");
}

void printIntArray(int *T, int n)
{
    int i;
    for (i = 0; i < n; i++)
        printf("%d ", T[i]);
    puts("");
}

// mult with thread
int alg_matmul2D(float *a, float *b, float *c, int n)
{
    int i, j, k;
    double tCompStart = MPI_Wtime();

#pragma omp parallel shared(a, b, c) private(i, j, k)
    {
#pragma omp for schedule(static)
        for (i = 0; i < n; i = i + 1)
        {
            for (j = 0; j < n; j = j + 1)
            {

                for (k = 0; k < n; k = k + 1)
                {
                    c[i * n + j] += a[i * n + k] * b[k * n + j];
                }
            }
        }
    }
    return 0;
}

int main(int argc, char *argv[])
{
    int rank, size, colID, rowID, ndim;
    MPI_Comm comm;
    int i = atoi(argv[1]); // matris dimension sayısı dışarıdan parametre olarak alınır
    int j = 0, l;
    int *sendcount;

    // veri dağımında scatter için kullanıcak disp verilerini ayarlamak için
    int *disp;
    int k;
    int z = 0;
    
    // kartezyen topolojli oluşturma bilgileri
    int dim[2], period[2], reorder;
    int coords[2] ,id;

    // master da kullanılacak A B C ve local matrisler
    float *matrixA, *matrixB, *matrixC, *localA, *localB, *localC, *tempA, *tempB;

    // time hesaplama değişkenleri
    double timeCommStart, timeCommEnd, timeCompStart, timeCompEnd, timeWallStart, timeWallEnd;
    double totalCompTime = 0.0, totalCommTime = 0.0;

    //MPI başlatır
    MPI_Init(&argc, &argv);

    timeWallStart = MPI_Wtime(); // time start

    // rank ve size bilgilerini çek
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // her bir processte bulunacak local matrixlerin tek boyutunu hesaplar
    int matrixOneSize = sqrt((i * i) / size);
    // her bir processte ki local matrixlerin boyutunu hesaplar
    int localMatrixSize = matrixOneSize * matrixOneSize;

    // matrixA ve matrixB yi prosesslere düzenli şekilde dağıtmak için displacement matrix ini oluşturur
    disp = create1DIntArray(size);
    for (k = 0; k < i / matrixOneSize; k++)
    {
        j += (i * matrixOneSize) * k;
        for (l = 0; l < i / matrixOneSize; l++)
        {
            disp[z] = j;
            z = z + 1;
            j = j + matrixOneSize;
        }
        j = 0;
    }

    // gönderilecek matrix boyutunu scatter içerisinde tanımlamak için kullanılır
    sendcount = create1DIntArray(size);
    fillArrayWithOne(sendcount, size);

    // cartesian topoloji özellik atamaları yapar
    ndim = 2;
    dim[0] = sqrt(size);
    dim[1] = sqrt(size);
    period[0] = 1;
    period[1] = 0;
    reorder = 1;

    // caretsian topolojiyi oluşturur
    MPI_Cart_create(MPI_COMM_WORLD, ndim, dim, period, reorder, &comm);
    MPI_Comm_rank(comm, &id);
    MPI_Cart_coords(comm, id, ndim, coords);
    MPI_Barrier(MPI_COMM_WORLD);

    // local process lerde ihtiyacımız olan matrix leri oluşturur
    localA = create1DArray(localMatrixSize);
    localB = create1DArray(localMatrixSize);
    localC = create1DArray(localMatrixSize);
    tempA = create1DArray(localMatrixSize);
    tempB = create1DArray(localMatrixSize);

    // localC matrix i üzerinde toplama işlemi yapılacağından tüm processlerde ilk değer ataması yapar
    fillArrayWithZero(localC, localMatrixSize);

    if (rank == MASTER)
    {
        // dağıtılacak olan A ve B matrixlerini doldurur
        matrixA = create1DArray(i * i);
        fillArray(matrixA, i * i);

        matrixB = create1DArray(i * i);
        fillArray(matrixB, i * i);

        matrixC = create1DArray(i * i);
    }

    // tüm processler ana process in doldurma işlemini tamamlamasını bekler
    MPI_Barrier(MPI_COMM_WORLD);

    // ana matrixler process sayısına göre parçalanacağından yeni bir datatype oluşturur
    MPI_Datatype matrixType, newMatrixType;
    int blocklength = sqrt((i * i) / size), stride = i;
    MPI_Type_vector(blocklength, blocklength, stride, MPI_FLOAT, &matrixType);
    MPI_Type_commit(&matrixType);

    MPI_Type_create_resized(matrixType, 0, 1 * sizeof(float), &newMatrixType);
    MPI_Type_commit(&newMatrixType);

    timeCommStart = MPI_Wtime(); // time start
    // Scatter ile oluşturulan datatype a uygun şekilde ana matrixleri parçalayarak dağıtır
    MPI_Scatterv(matrixA, sendcount, disp, newMatrixType, localA, localMatrixSize, MPI_FLOAT, MASTER, comm);
    MPI_Scatterv(matrixB, sendcount, disp, newMatrixType, localB, localMatrixSize, MPI_FLOAT, MASTER, comm);
    timeCommEnd = MPI_Wtime(); // time end

    // toplam haberleşme süresine ekle
    totalCommTime += timeCommEnd - timeCommStart;

    // ### çarpılacak satır ve sütünları dağıtma işlemleri ###

    // A ve B matrixlerini sıralı şekilde gönderip almak için farklı send recv tagları kullan
    int tagA = 777;
    int tagB = 778;

    // veri gönderim rankları
    int sendRank;
    int sendCoords[2];
    
    // verinin alıncağı ranklar
    int receiveRank;
    int receiveCoords[2];

    //MPI_Irsend ve Irecv işlemlerinin tamamlama bilgilerini yakalayacak değişkenler
    MPI_Request request1, request2, request3, request4;
    MPI_Status status;

    // çarpılcak satır ve sütünlerı ilgili processlere gönderir
    for (int p = 0; p < sqrt(size); p++)
    {
        //rankı localA yı göndermek için ayarla
        sendCoords[0] = coords[0];
        sendCoords[1] = p;
        MPI_Cart_rank(comm, sendCoords, &sendRank);

        timeCommStart = MPI_Wtime(); // time start
        //localA yı ilgili ranka yolla
        MPI_Irsend(localA, localMatrixSize, MPI_FLOAT, sendRank, tagA,
                   comm, &request1);
        //MPI_Wait(&request1,&status);

        //rankı lacalB yi göndermek için ayarla
        sendCoords[0] = p;
        sendCoords[1] = coords[1];
        MPI_Cart_rank(comm, sendCoords, &sendRank);

        // localB yi ilgili ranka yolla
        MPI_Irsend(localB, localMatrixSize, MPI_FLOAT, sendRank, tagB,
                   comm, &request2);
        timeCommEnd = MPI_Wtime(); // time end

        totalCommTime += timeCommEnd - timeCommStart;
    }

    // gönderme işlemini tüm processlerin tamamlamasını bekle
    MPI_Barrier(MPI_COMM_WORLD);

    // gönderilen satır ve sütunlerı ilgili matrise toplar ve hesaplama işlemini yaptırır
    for (int o = 0; o < sqrt(size); o++)
    {
        //localA için alıcı ranklarını ayarla
        receiveCoords[0] = coords[0];
        receiveCoords[1] = o;
        MPI_Cart_rank(comm, receiveCoords, &receiveRank);

        timeCommStart = MPI_Wtime(); // time start
        //localA yı ilgili ranka yolla
        MPI_Irecv(tempA, localMatrixSize, MPI_FLOAT, receiveRank, tagA,
                  comm, &request3);

        //localB için alırı ranklarını ayarla
        receiveCoords[0] = o;
        receiveCoords[1] = coords[1];
        MPI_Cart_rank(comm, receiveCoords, &receiveRank);

        //localB yi ilgili ranka yolla
        MPI_Irecv(tempB, localMatrixSize, MPI_FLOAT, receiveRank, tagB,
                  comm, &request4);
        timeCommEnd = MPI_Wtime(); // time end

        totalCommTime += timeCommEnd - timeCommStart;
        timeCompStart = MPI_Wtime(); // time start
        // tempA ve tempB matrixlerini çarpıp, localC deki matrix değerleri ile toplar
        alg_matmul2D(tempA, tempB, localC, matrixOneSize);
        timeCompEnd = MPI_Wtime(); // time end

        // total hesaplama süresine ekle
        totalCompTime += timeCompEnd - timeCompStart;
    }

    timeCommStart = MPI_Wtime(); // time start
    // tüm localC leri master process e toplar
    MPI_Gatherv(localC, localMatrixSize, MPI_FLOAT, matrixC, sendcount, disp, newMatrixType, MASTER, comm);
    timeCommEnd = MPI_Wtime(); // time end

    // total haberleşme süresine ekle
    totalCommTime += timeCommEnd - timeCommStart;

    timeWallEnd = MPI_Wtime(); // tüm program çalışmayı bitirdi

    if (rank == MASTER)
    {
        printf(" %d   %d   --Calculation End--\n", size, i);
        printf("Communication Time = %f sec.\n", totalCommTime);
        printf("Computation Time = %f sec.\n", totalCompTime);
        printf("Total Time = %f sec.\n", timeWallEnd - timeWallStart);
    }

    // MPI bitir
    MPI_Finalize();

    return 0;
}
