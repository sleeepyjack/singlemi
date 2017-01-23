#ifndef EXHAUSTIVE_SEARCH_CUH
#define EXHAUSTIVE_SEARCH_CUH
#include "cudahelpers/cuda_helpers.cuh"
#include "alg515/alg_515.cuh"

template <typename index_t>
HOSTDEVICEQUALIFIER INLINEQUALIFIER index_t ipow(index_t base, index_t expo)
{
	index_t result = 1;
	while(expo)
	{
		if (expo & 1)
			result *= base;
		expo >>= 1;
		base *= base;
	}
	return result;
}

template <typename index_t>
HOSTDEVICEQUALIFIER INLINEQUALIFIER index_t next_genotype(index_t i, index_t k) //there are pow(3, k)*k distinct combinations
{
	return ((i/k)/ipow(index_t(3), i % k)) % 3;
}

template <
	typename index_t,
	typename array_t,
	typename value_t,
	index_t MAX_K = 6
	>
GLOBALQUALIFIER void exhaustive_search_kernel(array_t * cases,
																							index_t sizeCases,
																							array_t * ctrls,
																							index_t sizeCtrls,
																							index_t * candidates,
																							index_t n,
																							index_t k,
																						 	index_t numCombinations,
																						 	value_t entY,
																						 	value_t invInds,
																						 	value_t * mi)
{
	extern __shared__ index_t geno[];
	//init shared
	for(index_t thid = threadIdx.x; thid < ipow(index_t(3), k)*k; thid += blockDim.x)
	{
		if(thid >= ipow(index_t(3), k)*k) break;
		geno[thid] = next_genotype(thid, k);
	}
	index_t comb[MAX_K];
	array_t intersectCases, intersectCtrls;
	index_t popCntCases, popCntCtrls;
	value_t entX, entAll, pCases, pCtrls;
	__syncthreads();
	// grid stride loop

	for(index_t thid = blockIdx.x * blockDim.x + threadIdx.x; thid < numCombinations; thid += blockDim.x*gridDim.x)
	{
		if(thid >= numCombinations) return;

		entX   = 0.0;
		entAll = 0.0;

		alg515(n, k, thid, comb); // get SNP combination from thid

		// substitute candidate IDs for combination
		#pragma unroll MAX_K
		for (index_t i = 0; i < k; i++) {
			comb[i] = candidates[comb[i]];
		}

		// for each genotype combination
		// TODO unroll?
		for(index_t g = 0; g < ipow(index_t(3), k); g++)
		{
			popCntCases = 0;
			popCntCtrls = 0;

			// CASES
			// for each 32-sample batch
			for(index_t i = 0; i < sizeCases; i++)
			{
				intersectCases = {~(array_t(0))}; // identity with &. e.g.: 0xFFFFFFFF

				// for each snp in the combination
				for(index_t j = 0; j < k; j++)
				{
					intersectCases &= cases[(3 * sizeCases * comb[j])
					+(sizeCases
					* geno[g*k+j])+i]; // indexing mayhem: cases[(snp)+(genotype)+i]
				}
				popCntCases += __popc(intersectCases);
			}
			// CTRLS
			// for each 32-sample batch
			for(index_t i = 0; i < sizeCtrls; i++)
			{
				intersectCtrls = {~(array_t(0))}; // identity with &. e.g.: 0xFFFFFFFF

				// for each snp in the combination
				for(index_t j = 0; j < k; j++)
				{
					intersectCtrls &= ctrls[(3 * sizeCtrls * comb[j])+(sizeCtrls * geno[g*k+j])+i]; // indexing mayhem: ctrls[(snp)+(genotype)+i]
				}
				popCntCtrls += __popc(intersectCtrls);
			}
			// calculate entropies
			pCases = popCntCases * invInds;
			if(pCases != 0.0) entAll -= pCases * ::log2(pCases); // NOTE log2 ?

			pCtrls = popCntCtrls * invInds;
			if(pCtrls != 0.0) entAll -= pCtrls * ::log2(pCtrls);

			pCases += pCtrls;
			if(pCases != 0.0) entX   -= pCases * ::log2(pCases);
		}
		mi[thid] = (entX + entY - entAll); // definition of mutual information
	}
}

#endif /* EXHAUSTIVE_SEARCH_CUH */
