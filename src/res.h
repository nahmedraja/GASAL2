#define HOST (false)
#define DEVICE (true)


gasal_res_t *gasal_res_new(uint32_t max_n_alns, Parameters *params, bool device = false);
void gasal_res_destroy(gasal_res_t *res) ;
