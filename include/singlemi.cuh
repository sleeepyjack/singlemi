#ifndef SINGLEMI_CUH
#define SINGLEMI_CUH

template <
  typename index_t,
  typename array_t,
  typename value_t
  >
GLOBALQUALIFIER void singlemi_kernel( array_t * cases,
  																		index_t sizeCases,
  																		array_t * ctrls,
  																		index_t sizeCtrls,
  																		index_t n,
  																	 	value_t H_y,
  																	 	value_t invInds,
  																	 	value_t * mi)
{
  index_t f_cases, f_ctrls;
  value_t p_cases, p_ctrls;
  value_t H_x, H_xy;

  for(index_t thid = blockIdx.x*blockDim.x + threadIdx.x; thid < n; thid += blockDim.x * gridDim.x)
  {
      H_x = 0.0;
      H_xy = 0.0;

      for(index_t geno = 0; geno < 3; geno++)
      {
          f_cases = 0;
          f_ctrls = 0;
          for(index_t i = (3*thid+geno)*sizeCases; i < (3*thid+1+geno)*sizeCases; i++)
          {
              f_cases += __popc(cases[i]);
          }
          for(index_t i = (3*thid+geno)*sizeCtrls; i < (3*thid+1+geno)*sizeCtrls; i++)
          {
              f_ctrls += __popc(ctrls[i]);
          }

          // calculate entropies
    			p_cases = f_cases*invInds;

    			if(p_cases != 0.0) H_xy -= p_cases * ::log2(p_cases); // NOTE log2 ?

    			p_ctrls = f_ctrls*invInds;
    			if(p_ctrls != 0.0) H_xy -= p_ctrls * ::log2(p_ctrls);

    			p_cases += p_ctrls;
    			if(p_cases != 0.0) H_x  -= p_cases * ::log2(p_cases);
      }

      mi[thid] = H_x + H_y - H_xy;
  }
}


#endif /* SINGLEMI_CUH */
