#include <iostream>
#include "include/cudahelpers/cuda_helpers.cuh"
#include "include/cxxopts/include/cxxopts.hpp"
#include "snpreader/SNPReader.h"
#include "include/singlemi.cuh"
#include "include/cub/cub/device/device_radix_sort.cuh"
#include "include/alg515/alg_515.cuh"
#include "include/exhaustive_search.cuh"

template <
  typename T,
  typename index_t
  >
GLOBALQUALIFIER void iota_kernel(T * arr, index_t size)
{
  for(index_t thid = blockIdx.x*blockDim.x + threadIdx.x; thid < size; thid += blockDim.x * gridDim.x)
  {
    if(thid < size) arr[thid] = thid;
  }
}

template <
  typename index_t,
  typename array_t
  >
void convert_data(SNPReader* reader, array_t* cases, index_t sizeCases, array_t* ctrls, index_t sizeCtrls)
{
  vector<SNP*> snps = reader->getSnpSet();

  for(index_t i=0; i < reader->getNumSnp(); i++)
  {
    memcpy(cases+(i*3*sizeCases)+(0*sizeCases), snps[i]->_case0Values, sizeof(array_t)*sizeCases);
    memcpy(cases+(i*3*sizeCases)+(1*sizeCases), snps[i]->_case1Values, sizeof(array_t)*sizeCases);
    memcpy(cases+(i*3*sizeCases)+(2*sizeCases), snps[i]->_case2Values, sizeof(array_t)*sizeCases);

    memcpy(ctrls+(i*3*sizeCtrls)+(0*sizeCtrls), snps[i]->_ctrl0Values, sizeof(array_t)*sizeCtrls);
    memcpy(ctrls+(i*3*sizeCtrls)+(1*sizeCtrls), snps[i]->_ctrl1Values, sizeof(array_t)*sizeCtrls);
    memcpy(ctrls+(i*3*sizeCtrls)+(2*sizeCtrls), snps[i]->_ctrl2Values, sizeof(array_t)*sizeCtrls);
  }
}

/*

template <
  typename index_t,
  typename array_t,
  typename value_t
  >
void filter(array_t* cases, index_t sizeCases, array_t* ctrls, index_t sizeCtrls, index_t n, index_t c, value_t Hy, value_t invInds, index_t* candidates)
{
  //evaluate I(X;Y) for each SNP X
  value_t * mi; cudaMalloc(&mi, sizeof(value_t)*n); CUERR
  singlemi_kernel<<<SDIV(n, BLOCKSIZE), BLOCKSIZE>>>(cases, sizeCases, ctrls, sizeCtrls, n, Hy, invInds, mi); CUERR

  //sort SNPs descending according to I(X;Y)
  index_t* tmp_index; cudaMalloc(&tmp_index, sizeof(index_t)*n); CUERR
  value_t* tmp_value; cudaMalloc(&tmp_value, sizeof(value_t)*n); CUERR
  iota_kernel<<<SDIV(n, BLOCKSIZE), BLOCKSIZE>>>(tmp_index, n); CUERR
  void    *cub_tmp = NULL;
  size_t  cub_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(cub_tmp, cub_bytes, mi, tmp_value, tmp_index, candidates, n); CUERR
  cudaMalloc(&cub_tmp, cub_bytes); CUERR
  cub::DeviceRadixSort::SortPairsDescending(cub_tmp, cub_bytes, mi, tmp_value, tmp_index, candidates, n); CUERR

  cudaFree(tmp_index); CUERR
  cudaFree(tmp_value); CUERR
  cudaFree(cub_tmp); CUERR
  cudaFree(mi); CUERR
}

template <
  typename index_t,
  typename array_t,
  typename value_t
  >
void exhaustive_search(array_t* cases, index_t sizeCases, array_t* ctrls, index_t sizeCtrls, index_t* candidates, index_t c, index_t k, value_t Hy, value_t invInds, index_t* combination_out, value_t* mi_out)
{
  //evaluate I(X_1, ..., X_k;Y) for each k-combination of candidate SNPs (exhaustive search)
  index_t nCk = binom(c, k);
  value_t* mi; cudaMalloc(&mi, sizeof(value_t)*nCk); CUERR
  exhaustive_search_kernel<<<SDIV((nCk < (1UL << 32)) ? nCk : (1UL << 32), BLOCKSIZE), BLOCKSIZE, (ipow(index_t(3), index_t(k))*k)*sizeof(index_t)>>>(cases, sizeCases, ctrls, sizeCtrls, candidates, c, k, nCk, Hy, invInds, mi); CUERR

  //sort combinations descending according to I(X_1, ..., X_k;Y)
  index_t* tmp_index; cudaMalloc(&tmp_index, sizeof(index_t)*nCk); CUERR
  iota_kernel<<<SDIV(n, BLOCKSIZE), BLOCKSIZE>>>(combination_out, nCk); CUERR
  void *cub_tmp = NULL;
  size_t cub_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(cub_tmp, cub_bytes, mi, mi_out, tmp_index, combination_out, nCk); CUERR
  cudaMalloc(&cub_tmp, cub_bytes); CUERR
  cub::DeviceRadixSort::SortPairsDescending(cub_tmp, cub_bytes, mi_d, tmp_value, tmp_index, combinations_d, nCk); CUERR
  combinations_h = (index_t*)malloc(sizeof(index_t)*nCk);
  cudaMemcpy(combinations_h, combinations_d, sizeof(index_t)*nCk, D2H); CUERR
  mi_h = (value_t*)malloc(sizeof(value_t)*nCk);
  cudaMemcpy(mi_h, tmp_value, sizeof(value_t)*nCk, D2H); CUERR
  cudaFree(combinations_d); CUERR
  cudaFree(mi_d); CUERR
  cudaFree(tmp_index); CUERR
  cudaFree(tmp_value); CUERR
  cudaFree(cub_tmp); CUERR
}
*/
int main(int argc, char *argv[]) {

  typedef uint32_t index_t;
  typedef uint32_t array_t;
  typedef float    value_t;

  constexpr index_t BLOCKSIZE = 256;

  std::string tped, tfam;
  index_t n, c, k, g;
  value_t H_y, invInds;
  index_t numCases, numCtrls, sizeCases, sizeCtrls, nCk;
  array_t *cases_h, *cases_d, *ctrls_h, *ctrls_d;
  index_t *candidates_h, *candidates_d, *combinations_h, *combinations_d, *tmp_index;
  value_t *mi_h, *mi_d, *tmp_value;

  //parse input parameters
  cxxopts::Options options(argv[0], "SingleMI - Ultra-Fast Detection of Higher-Order Epistatic Interactions on GPUs");
  options.add_options()
    ("p,tped", "TPED filename", cxxopts::value<std::string>(), "STRING")
    ("f,tfam", "TFAM filename", cxxopts::value<std::string>(), "STRING")
    ("c,candidates", "Number of SNPs considered for the exhaustive step", cxxopts::value<index_t>(), "INT")
    ("k,order", "Order of interaction", cxxopts::value<index_t>(), "INT")
    ("g,gpu", "Device ID of GPU to use for execution", cxxopts::value<index_t>()->default_value("0"), "INT")
    ("h,help", "Print help");
    options.parse(argc, argv);
    if (options.count("help"))
    {
      std::cout << options.help({""}) << std::endl;
      exit(0);
    }
    if (options.count("tped"))
    {
      tped = options["tped"].as<std::string>();
    }else{
      std::cerr << "ERROR: TPED file not specified" << std::endl;
      std::cout << options.help({""}) << std::endl;
      exit(1);
    }
    if (options.count("tfam"))
    {
      tfam = options["tfam"].as<std::string>();
    }else{
      std::cerr << "ERROR: TFAM file not specified" << std::endl;
      std::cout << options.help({""}) << std::endl;
      exit(1);
    }
    if (options.count("candidates")) {
      c   = options["candidates"].as<index_t>();
    }else{
      std::cerr << "ERROR: Number of candidates not specified" << std::endl;
      std::cout << options.help({""}) << std::endl;
      exit(1);
    }
    if (options.count("order"))
    {
      k = options["order"].as<index_t>();
      if (k < 2) {
        std::cerr << "ERROR: Order must be greater than 1" << std::endl;
        std::cout << options.help({""}) << std::endl;
        exit(1);
      }
    }else{
      std::cerr << "ERROR: Order of interaction not specified" << std::endl;
      std::cout << options.help({""}) << std::endl;
      exit(1);
    }
    g   = options["gpu"].as<index_t>();

    cudaSetDevice(g); CUERR

    SNPReader * reader = new SNPReader(tped.c_str(), tfam.c_str());

    reader->loadSNPSet();
    n = reader->getNumSnp();
    if (n < c) {
      std::cerr << "ERROR: Number of candidates cannot be greater than number of SNPs" << std::endl;
      std::cout << options.help({""}) << std::endl;
      exit(1);
    }
    numCases  = reader->getNumCases();
    numCtrls  = reader->getNumCtrls();
    sizeCases = SDIV(numCases, 32);
    sizeCtrls = SDIV(numCtrls, 32);
    H_y      = (-1.0)*((1.0*numCases/(numCases+numCtrls))*std::log2(1.0*numCases/(numCases+numCtrls))
                     +(1.0*numCtrls/(numCases+numCtrls))*std::log2(1.0*numCtrls/(numCases+numCtrls)));
    invInds   = (1.0/(numCases+numCtrls));
    nCk       = binom(c, k);

    cases_h = (array_t*)malloc(sizeof(array_t)*3*sizeCases*n);
    ctrls_h = (array_t*)malloc(sizeof(array_t)*3*sizeCtrls*n);
    cudaMalloc(&cases_d, sizeof(array_t)*3*sizeCases*n); CUERR
    cudaMalloc(&ctrls_d, sizeof(array_t)*3*sizeCtrls*n); CUERR

    convert_data(reader, cases_h, sizeCases, ctrls_h, sizeCtrls);

    cudaMemcpy(cases_d, cases_h, sizeof(array_t)*3*sizeCases*n, H2D); CUERR
    cudaMemcpy(ctrls_d, ctrls_h, sizeof(array_t)*3*sizeCtrls*n, H2D); CUERR


    //evaluate I(X;Y) for each SNP X
    cudaMalloc(&mi_d, sizeof(value_t)*n); CUERR
    singlemi_kernel<<<SDIV(n, BLOCKSIZE), BLOCKSIZE>>>(cases_d, sizeCases, ctrls_d, sizeCtrls, n, H_y, invInds, mi_d); CUERR

    //sort SNPs descending according to I(X;Y)
    cudaMalloc(&candidates_d, sizeof(index_t)*n); CUERR
    cudaMalloc(&tmp_index, sizeof(index_t)*n); CUERR
    cudaMalloc(&tmp_value, sizeof(value_t)*n); CUERR
    iota_kernel<<<SDIV(n, BLOCKSIZE), BLOCKSIZE>>>(tmp_index, n); CUERR
    void    *cub_tmp = NULL;
    size_t  cub_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(cub_tmp, cub_bytes, mi_d, tmp_value, tmp_index, candidates_d, n); CUERR
    cudaMalloc(&cub_tmp, cub_bytes); CUERR
    cub::DeviceRadixSort::SortPairsDescending(cub_tmp, cub_bytes, mi_d, tmp_value, tmp_index, candidates_d, n); CUERR
    candidates_h = (index_t*)malloc(sizeof(index_t)*c);
    cudaMemcpy(candidates_h, candidates_d, sizeof(index_t)*c, D2H); CUERR

    mi_h = (value_t*)malloc(sizeof(value_t)*n);
    cudaMemcpy(mi_h, tmp_value, sizeof(value_t)*n, D2H); CUERR
    std::cout.precision(4);
    std::cout << "####################################################" << std::endl;
    std::cout << "Selected Candidates:" << std::endl;
    std::cout << "ID\tMI" << std::endl;
    for (index_t i = 0; i < c; i++) {
      std::cout << candidates_h[i] << "\t" << mi_h[i] << std::endl;
    }

    cudaFree(tmp_index); CUERR
    cudaFree(tmp_value); CUERR
    cudaFree(cub_tmp); CUERR
    cudaFree(mi_d); CUERR


    //evaluate I(X_1, ..., X_k;Y) for each k-combination of candidate SNPs (exhaustive search)
    cudaMalloc(&mi_d, sizeof(value_t)*nCk); CUERR
    exhaustive_search_kernel<<<SDIV((nCk < (1UL << 32)) ? nCk : (1UL << 32), BLOCKSIZE), BLOCKSIZE, (ipow(index_t(3), index_t(k))*k)*sizeof(index_t)>>>(cases_d, sizeCases, ctrls_d, sizeCtrls, candidates_d, c, k, nCk, H_y, invInds, mi_d); CUERR
    cudaFree(candidates_d); CUERR

    //sort combinations descending according to I(X_1, ..., X_k;Y)
    cudaMalloc(&combinations_d, sizeof(index_t)*nCk); CUERR
    cudaMalloc(&tmp_index, sizeof(index_t)*nCk); CUERR
    cudaMalloc(&tmp_value, sizeof(value_t)*nCk); CUERR
    iota_kernel<<<SDIV(n, BLOCKSIZE), BLOCKSIZE>>>(tmp_index, nCk); CUERR
    cub_tmp = NULL;
    cub_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(cub_tmp, cub_bytes, mi_d, tmp_value, tmp_index, combinations_d, nCk); CUERR
    cudaMalloc(&cub_tmp, cub_bytes); CUERR
    cub::DeviceRadixSort::SortPairsDescending(cub_tmp, cub_bytes, mi_d, tmp_value, tmp_index, combinations_d, nCk); CUERR
    combinations_h = (index_t*)malloc(sizeof(index_t)*nCk);
    cudaMemcpy(combinations_h, combinations_d, sizeof(index_t)*nCk, D2H); CUERR
    mi_h = (value_t*)malloc(sizeof(value_t)*nCk);
    cudaMemcpy(mi_h, tmp_value, sizeof(value_t)*nCk, D2H); CUERR
    cudaFree(combinations_d); CUERR
    cudaFree(mi_d); CUERR
    cudaFree(tmp_index); CUERR
    cudaFree(tmp_value); CUERR
    cudaFree(cub_tmp); CUERR

    std::cout << "####################################################" << std::endl;
    std::cout << "Top 10 interactions:" << std::endl;
    std::cout << "ID";
    for (size_t i = 0; i < k; i++) {
      std::cout << "\t";
    }
    std::cout << "MI" << std::endl;
    index_t * comb = (index_t*)malloc(sizeof(index_t)*k);
    for (index_t i = 0; i < 10; i++) {
      alg515(c, k, combinations_h[i], comb);
      for (index_t j = 0; j < k; j++) {
  			std::cout << candidates_h[comb[j]] << "\t";
  		}
      std::cout << mi_h[i] << std::endl;
    }

    cudaDeviceSynchronize(); CUERR

}
