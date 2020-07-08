import sys
import numpy as np
import pandas as pd
import time
from sklearn.decomposition import TruncatedSVD
from iterative_svd import IterativeSVD
import matplotlib.pyplot as plt
import plotly_express as px
import plotly
from distutils.util import strtobool
from file_processing import get_masked_matrix, process_labels_weights, center_masked_matrix

def run_iterative_svd(X_incomplete, start_rank, end_rank, rank, choose_best, num_cores, save_completed_matrix, completed_matrix_filename):
    num_masked = np.isnan(X_incomplete).sum()
    if num_masked > 0:
        start_time = time.time()
        X_complete = IterativeSVD(start_rank=start_rank, end_rank=end_rank, rank=rank, choose_best=choose_best, num_cores=num_cores).fit_transform(X_incomplete)
        print("Iterative SVD --- %s seconds ---" % (time.time() - start_time))
    else:
        X_complete = X_incomplete
    if save_completed_matrix:
        np.save(completed_matrix_filename, X_complete)
    return X_complete

def weight_completed_matrix(X_complete, X_incomplete):
    num_samples, num_pos = X_complete.shape
    weights = np.sum(~np.isnan(X_incomplete), axis=1) / num_pos
#     weights = np.sqrt(weights)
    sum_weights = sum(weights)
    W = np.diag(weights) / sum_weights
    X_complete -= np.sum(np.matmul(W, X_complete), axis=0)
    Wsqrt_X = np.matmul(np.sqrt(W), X_complete)
    return Wsqrt_X, X_complete

def project_weighted_matrix(Wsqrt_X, X_complete):
    svd = TruncatedSVD(2, algorithm="arpack")
    WX_normalized = Wsqrt_X #- np.sum(WX, axis=0)
    svd.fit(WX_normalized)
    X_projected = svd.transform(X_complete)
    num_samples, num_pos = X_complete.shape
    total_var = np.trace(np.matmul(X_complete, X_complete.T)) / num_samples
    pc1_percentvar = 100 * np.var(X_projected[:,0]) / total_var
    pc2_percentvar = 100 * np.var(X_projected[:,1]) / total_var
    return X_projected, pc1_percentvar, pc2_percentvar

def scatter_plot(X_projected, scatterplot_filename, output_filename, ind_IDs, labels):
    plot_df = pd.DataFrame()
    plot_df['x'] = X_projected[:,0]
    plot_df['y'] = X_projected[:,1]
    plot_df['Label'] = labels
    plot_df['ID'] = ind_IDs
    scatter = px.scatter(plot_df, x='x', y='y', color='Label', hover_name='ID', color_discrete_sequence=px.colors.qualitative.Alphabet)
    plotly.offline.plot(scatter, filename = scatterplot_filename, auto_open=False)
    plot_df.to_csv(output_filename, columns=['ID', 'x', 'y'], sep='\t', index=False)
    
def run_method(beagle_or_vcf, beagle_filename, vcf_filename, is_masked, vit_or_fbk_or_tsv, vit_filename, fbk_filename, fb_or_msp, tsv_filename, num_ancestries, ancestry, prob_thresh, average_parents, start_rank, end_rank, rank, choose_best, num_cores, is_weighted, labels_filename, output_filename, scatterplot_filename, save_masked_matrix, masked_matrix_filename, save_completed_matrix, completed_matrix_filename):
    X_incomplete, ind_IDs, rs_IDs = get_masked_matrix(beagle_filename, vcf_filename, beagle_or_vcf, is_masked, vit_filename, fbk_filename, tsv_filename, vit_or_fbk_or_tsv, fb_or_msp, num_ancestries, ancestry, average_parents, prob_thresh)
    X_incomplete, ind_IDs, labels, _ = process_labels_weights(labels_filename, X_incomplete, ind_IDs, average_parents, is_weighted, save_masked_matrix, masked_matrix_filename)
    X_incomplete = center_masked_matrix(X_incomplete)
    X_complete = run_iterative_svd(X_incomplete, start_rank, end_rank, rank, choose_best, num_cores, save_completed_matrix, completed_matrix_filename)
    Wsqrt_X, X_complete = weight_completed_matrix(X_complete, X_incomplete)
    X_projected, pc1_percentvar, pc2_percentvar = project_weighted_matrix(Wsqrt_X, X_complete)
    scatter_plot(X_projected, scatterplot_filename, output_filename, ind_IDs, labels)
    print("Percent variance explained by the 1st principal component: ", pc1_percentvar)
    print("Percent variance explained by the 2nd principal component: ", pc2_percentvar)

def run(params_filename):
    file = open(params_filename)
    params = {}
    for line in file:
        line = line.strip()
        if not line.startswith('#'):
            key_value = line.split('=')
            if len(key_value) == 2:
                params[key_value[0].strip()] = key_value[1].strip()
    file.close()
    beagle_or_vcf = int(params['BEAGLE_OR_VCF'])
    beagle_filename = str(params['BEAGLE_FILE'])
    vcf_filename = str(params['VCF_FILE'])
    is_masked = bool(strtobool(params['IS_MASKED']))
    vit_or_fbk_or_tsv = int(params['VIT_OR_FBK_OR_TSV'])
    vit_filename = str(params['VIT_FILE'])
    fbk_filename = str(params['FBK_FILE'])
    fb_or_msp = int(params['FB_OR_MSP'])
    tsv_filename = str(params['TSV_FILE'])
    num_ancestries = int(params['NUM_ANCESTRIES'])
    ancestry = int(params['ANCESTRY'])
    prob_thresh = float(params['PROB_THRESH'])
    average_parents = bool(strtobool(params['AVERAGE_PARENTS']))
    start_rank = int(params['START_RANK'])
    end_rank = int(params['END_RANK'])
    rank = int(params['RANK'])
    choose_best = bool(strtobool(params['CHOOSE_BEST']))
    num_cores = int(params['NUM_CORES'])
    is_weighted = bool(strtobool(params['IS_WEIGHTED']))
    labels_filename = str(params['LABELS_FILE'])
    output_filename = str(params['OUTPUT_FILE'])
    scatterplot_filename = str(params['SCATTERPLOT_FILE'])
    save_masked_matrix = bool(strtobool(params['SAVE_MASKED_MATRIX']))
    masked_matrix_filename = str(params['MASKED_MATRIX_FILE'])
    save_completed_matrix = bool(strtobool(params['SAVE_COMPLETED_MATRIX']))
    completed_matrix_filename = str(params['COMPLETED_MATRIX_FILE'])
    run_method(beagle_or_vcf, beagle_filename, vcf_filename, is_masked, vit_or_fbk_or_tsv, vit_filename, fbk_filename, fb_or_msp, tsv_filename, num_ancestries, ancestry, prob_thresh, average_parents, start_rank, end_rank, rank, choose_best, num_cores, is_weighted, labels_filename, output_filename, scatterplot_filename, save_masked_matrix, masked_matrix_filename, save_completed_matrix, completed_matrix_filename)
    
def main():
    params_filename = sys.argv[1]
    start_time = time.time()
    run(params_filename)
    print("Total time --- %s seconds ---" % (time.time() - start_time))
    
if __name__ == "__main__":
    main()