// gmm_analysis.c
// -------------------------------------------------------------
// Evaluates separation and cohesion of a trained GMM (.gmm file)
//
// COMPILING:
//   gcc -O2 tests/gmm_analysis.c -lm -o gmm_analysis
//
// RUNNING:
//   ./gmm_analysis gmm_indexes/gmm/coco-i2i-512-angular_train_reduced.gmm
//   ./gmm_analysis gmm_indexes/gmm/fashion-mnist-784-euclidean_train_reduced.gmm
//   ./gmm_analysis gmm_indexes/gmm/gist-960-euclidean_train_reduced.gmm
//   ./gmm_analysis gmm_indexes/gmm/glove-25-angular_train_reduced.gmm
//   ./gmm_analysis gmm_indexes/gmm/glove-50-angular_train_reduced.gmm
//   ./gmm_analysis gmm_indexes/gmm/glove-100-angular_train_reduced.gmm
//   ./gmm_analysis gmm_indexes/gmm/glove-200-angular_train_reduced.gmm
//   ./gmm_analysis gmm_indexes/gmm/lastfm-64-dot_train_reduced.gmm
//   ./gmm_analysis gmm_indexes/gmm/mnist-784-euclidean_train_reduced.gmm
//   ./gmm_analysis gmm_indexes/gmm/nytimes-256-angular_train_reduced.gmm
//   ./gmm_analysis gmm_indexes/gmm/sift-128-euclidean_train_reduced.gmm
//
// OUTPUT:
//   Prints cluster cohesion (variance), separation (mean inter-cluster distance),
//   and overall separation/cohesion ratio.
//
// -------------------------------------------------------------

/*
    ./gmm_analysis gmm_indexes/gmm/coco-i2i-512-angular_train_reduced.gmm
    ./gmm_analysis gmm_indexes/gmm/fashion-mnist-784-euclidean_train_reduced.gmm
    ./gmm_analysis gmm_indexes/gmm/gist-960-euclidean_train_reduced.gmm
    ./gmm_analysis gmm_indexes/gmm/glove-25-angular_train_reduced.gmm
    ./gmm_analysis gmm_indexes/gmm/glove-50-angular_train_reduced.gmm
    ./gmm_analysis gmm_indexes/gmm/glove-100-angular_train_reduced.gmm
    ./gmm_analysis gmm_indexes/gmm/glove-200-angular_train_reduced.gmm
    ./gmm_analysis gmm_indexes/gmm/lastfm-64-dot_train_reduced.gmm
    ./gmm_analysis gmm_indexes/gmm/mnist-784-euclidean_train_reduced.gmm
    ./gmm_analysis gmm_indexes/gmm/nytimes-256-angular_train_reduced.gmm
    ./gmm_analysis gmm_indexes/gmm/sift-128-euclidean_train_reduced.gmm
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#ifdef _WIN32
  #include <direct.h>
  #define MKDIR(p) _mkdir(p)
#else
  #include <sys/types.h>
  #include <sys/stat.h>
  #define MKDIR(p) mkdir(p, 0755)
#endif

typedef struct {
    uint32_t K;
    uint32_t dim;
    uint32_t metric;     // 0=Euclidean, 1=Cosine
    uint32_t normalised; // 0/1
    float var_floor;
} gmm_file_header;

static void l2_normalise(float* v, uint32_t dim) {
    double nrm=0.0;
    for(uint32_t d=0;d<dim;++d) nrm+=(double)v[d]*(double)v[d];
    nrm=sqrt(nrm);
    if(nrm>0.0){
        float inv=(float)(1.0/nrm);
        for(uint32_t d=0;d<dim;++d) v[d]*=inv;
    }
}

static float euclidean_distance(const float* a, const float* b, uint32_t dim) {
    double sum=0.0;
    for(uint32_t d=0; d<dim; ++d){
        double df=(double)a[d]-(double)b[d];
        sum+=df*df;
    }
    return (float)sqrt(sum);
}

static float cosine_distance(const float* a, const float* b, uint32_t dim) {
    double dot=0.0,n1=0.0,n2=0.0;
    for(uint32_t d=0;d<dim;++d){
        dot+=(double)a[d]*(double)b[d];
        n1+=(double)a[d]*(double)a[d];
        n2+=(double)b[d]*(double)b[d];
    }
    double denom=sqrt(n1)*sqrt(n2);
    if(denom==0.0) return 1.0f;
    double sim=dot/denom;
    return (float)(1.0 - sim);
}

int main(int argc, char** argv) {
    if(argc<2){
        fprintf(stderr,"Usage: %s <gmm_file>\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];
    FILE* f = fopen(path,"rb");
    if(!f){ perror("fopen"); return 1; }

    gmm_file_header h;
    fread(&h,sizeof(h),1,f);

    printf("Analysing GMM: %s\n", path);
    printf("  K=%u, dim=%u, metric=%s, normalised=%s\n",
           h.K,h.dim, h.metric?"cosine":"euclidean", h.normalised?"yes":"no");

    float* weights   = (float*)malloc(sizeof(float)*h.K);
    float* means     = (float*)malloc(sizeof(float)*h.K*h.dim);
    float* variances = (float*)malloc(sizeof(float)*h.K*h.dim);
    float* log_consts= (float*)malloc(sizeof(float)*h.K);
    if(!weights||!means||!variances||!log_consts){
        fprintf(stderr,"OOM.\n"); return 1;
    }

    fread(weights,sizeof(float),h.K,f);
    fread(means,sizeof(float),(size_t)h.K*h.dim,f);
    fread(variances,sizeof(float),(size_t)h.K*h.dim,f);
    fread(log_consts,sizeof(float),h.K,f);
    fclose(f);

    // ---- Compute cohesion ----
    float* coh = (float*)malloc(sizeof(float)*h.K);
    double total_coh=0.0;
    for(uint32_t k=0;k<h.K;++k){
        double sumv=0.0;
        for(uint32_t d=0;d<h.dim;++d)
            sumv += (double)variances[k*h.dim+d];
        coh[k] = (float)(sumv / (double)h.dim);
        total_coh += coh[k];
    }
    float avg_coh = (float)(total_coh / (double)h.K);

    // ---- Compute separation ----
    double total_sep=0.0;
    uint64_t pairs=0;
    for(uint32_t i=0;i<h.K;++i){
        for(uint32_t j=i+1;j<h.K;++j){
            float d;
            if(h.metric==0) d = euclidean_distance(&means[i*h.dim], &means[j*h.dim], h.dim);
            else            d = cosine_distance(&means[i*h.dim], &means[j*h.dim], h.dim);
            total_sep += (double)d;
            pairs++;
        }
    }
    float avg_sep = (pairs>0)? (float)(total_sep/(double)pairs):0.0f;
    float sep_coh_ratio = (avg_coh>0.0f)? (avg_sep/avg_coh) : 0.0f;

    // ---- Summary ----
    printf("\n=== GMM Separation–Cohesion Analysis ===\n");
    printf("Average cohesion (within-cluster variance): %.6f\n", avg_coh);
    printf("Average separation (between-cluster mean distance): %.6f\n", avg_sep);
    printf("Separation/Cohesion ratio: %.6f\n", sep_coh_ratio);
    printf("\nPer-cluster cohesion (variance mean per cluster):\n");
    for(uint32_t k=0;k<h.K;++k)
        printf("  Cluster %3u: mean variance = %.6f, weight = %.5f\n", k, coh[k], weights[k]);

    // optional: save to file
    char outpath[512];
    snprintf(outpath,sizeof(outpath),"analysis_results/%s_analysis.txt",
             strrchr(path,'/') ? strrchr(path,'/')+1 : path);
#ifdef _WIN32
    MKDIR("analysis_results");
#else
    mkdir("analysis_results",0755);
#endif
    FILE* out=fopen(outpath,"w");
    if(out){
        fprintf(out,"GMM Analysis for %s\n", path);
        fprintf(out,"K=%u dim=%u metric=%s\n",h.K,h.dim,h.metric?"cosine":"euclidean");
        fprintf(out,"Average cohesion=%.6f\nAverage separation=%.6f\nRatio=%.6f\n",
                avg_coh,avg_sep,sep_coh_ratio);
        for(uint32_t k=0;k<h.K;++k)
            fprintf(out,"Cluster %3u: mean variance=%.6f weight=%.5f\n",
                    k, coh[k], weights[k]);
        fclose(out);
    }

    printf("\nResults saved to: %s\n", outpath);

    free(weights); free(means); free(variances); free(log_consts); free(coh);
    return 0;
}