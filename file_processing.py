import allel
import numpy as np
import pandas as pd
import time
import sys

def process_vit(vit_file):
    vit_matrix = []
    with open(vit_file) as file:
        for x in file:
            x_split = x.replace('\n', '').split('\t')
            vit_matrix.append(np.array(x_split[1:-1]))
    ancestry_matrix = np.stack(vit_matrix, axis=0).T
    return ancestry_matrix

def process_fbk(fbk_file, num_ancestries, prob_thresh):
    df_fbk = pd.read_csv(fbk_file, sep=" ", header=None)
    fbk_matrix = df_fbk.values[:, :-1]
    ancestry_matrix = np.zeros((fbk_matrix.shape[0], int(fbk_matrix.shape[1] / num_ancestries)), dtype=np.int8)
    for i in range(num_ancestries):
        ancestry = i+1
        ancestry_matrix += (fbk_matrix[:, i::num_ancestries] > prob_thresh) * 1 * ancestry
    ancestry_matrix = ancestry_matrix.astype(str)
    return ancestry_matrix

def process_tsv_fb(tsv_file, num_ancestries, prob_thresh, positions, gt_matrix):
    df_tsv = pd.read_csv(tsv_file, sep="\t", skiprows=1)
    tsv_positions = df_tsv['physical_position'].tolist()
    df_tsv.drop(columns = ['physical_position', 'chromosome', 'genetic_position', 'genetic_marker_index'], inplace=True)
    tsv_matrix = df_tsv.values
    i_start = positions.index(tsv_positions[0])
    i_end = positions.index(tsv_positions[-1]) + 1
    gt_matrix = gt_matrix[i_start:i_end, :]
    positions = positions[i_start:i_end]
    prob_matrix = np.zeros((len(positions), tsv_matrix.shape[1]), dtype=np.float32)

    i_tsv = -1
    next_pos_tsv = tsv_positions[i_tsv+1]
    for i in range(len(positions)):
        pos = positions[i]
        if pos >= next_pos_tsv and i_tsv + 1 < tsv_matrix.shape[0]:
            i_tsv += 1
            probs = tsv_matrix[i_tsv, :]
            if i_tsv + 1 < tsv_matrix.shape[0]:
                next_pos_tsv = tsv_positions[i_tsv+1]
        prob_matrix[i, :] = probs

    tsv_matrix = prob_matrix
    ancestry_matrix = np.zeros((tsv_matrix.shape[0], int(tsv_matrix.shape[1] / num_ancestries)), dtype=np.int8)
    for i in range(num_ancestries):
        ancestry = i+1
        ancestry_matrix += (tsv_matrix[:, i::num_ancestries] > prob_thresh) * 1 * ancestry
    ancestry_matrix -= 1
    ancestry_matrix = ancestry_matrix.astype(str)
    return ancestry_matrix, gt_matrix

def process_tsv_msp(tsv_file, positions, gt_matrix):
    df_tsv = pd.read_csv(tsv_file, sep="\t", skiprows=1)
    tsv_spos = df_tsv['spos'].tolist()
    tsv_epos = df_tsv['epos'].tolist()
    df_tsv.drop(columns = ['#chm', 'spos', 'epos', 'sgpos', 'egpos', 'n snps'], inplace=True)
    tsv_matrix = df_tsv.values
    i_start = positions.index(tsv_spos[0])
    i_end = positions.index(tsv_epos[-1])
    gt_matrix = gt_matrix[i_start:i_end, :]
    positions = positions[i_start:i_end]
    ancestry_matrix = np.zeros((len(positions), tsv_matrix.shape[1]), dtype=np.int8)

    i_tsv = -1
    next_pos_tsv = tsv_spos[i_tsv+1]
    for i in range(len(positions)):
        pos = positions[i]
        if pos >= next_pos_tsv and i_tsv + 1 < tsv_matrix.shape[0]:
            i_tsv += 1
            ancs = tsv_matrix[i_tsv, :]
            if i_tsv + 1 < tsv_matrix.shape[0]:
                next_pos_tsv = tsv_spos[i_tsv+1]
        ancestry_matrix[i, :] = ancs

    ancestry_matrix = ancestry_matrix.astype(str)
    return ancestry_matrix, gt_matrix

def process_beagle(beagle_file):    
    rs_IDs = []
    lis_beagle = []
    with open(beagle_file) as file:
        x = file.readline()
        x_split = x.replace('\n', '').split('\t')
        ind_IDs = x_split[2:]
        ind_IDs = np.array(ind_IDs)
        for x in file:
            x_split = x.replace('\n', '').split('\t')
            if x_split[1][:2] == 'rs':
                rs_IDs.append(int(x_split[1][2:]))
            else:
                rs_ID_split = x_split[1].split('_')
                rs_IDs.append(np.float64(rs_ID_split[0] + '.' + rs_ID_split[1]))
            lis_beagle.append(x_split[2:])
    
    start_time = time.time()
    gt_matrix = np.zeros((len(lis_beagle),len(lis_beagle[0])), dtype=np.float32)
    
    for i in range(len(lis_beagle)):
        ref = lis_beagle[i][0]
        for j in range(1, len(lis_beagle[i])):
            gt_matrix[i, j] = (lis_beagle[i][j] != ref)*1

    print("Beagle Encoding Time: --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    
    return gt_matrix, ind_IDs, rs_IDs

def process_vcf(vcf_file):
    vcf = allel.read_vcf(vcf_file)
    gt = vcf['calldata/GT']
    n_variants, n_samples, ploidy = gt.shape
    gt_matrix = gt.reshape(n_variants, n_samples * ploidy).astype(np.float32)
    np.place(gt_matrix, gt_matrix < 0, np.nan)
    IDs = vcf['variants/ID']
    rs_IDs = [int(x[2:]) for x in IDs]
    samples = vcf['samples']
    ind_IDs = []
    for sample in samples:
        ind_IDs.append(sample + '_A')
        ind_IDs.append(sample + '_B')
    ind_IDs = np.array(ind_IDs)
    positions = vcf['variants/POS'].tolist()
    return gt_matrix, rs_IDs, ind_IDs, positions

def mask(ancestry_matrix, gt_matrix, unique_ancestries):
    start_time = time.time()
    masked_matrices = {}
    for ancestry in unique_ancestries:
        masked = np.empty(ancestry_matrix.shape[0] * ancestry_matrix.shape[1], dtype=np.float32)
        masked[:] = np.NaN
        arg = np.argwhere(ancestry_matrix.reshape(-1) == ancestry)
        masked[arg] = gt_matrix.reshape(-1)[arg]
        masked_matrices[ancestry] = masked.reshape(ancestry_matrix.shape)
        print("Masking for ancestry --- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
    return masked_matrices

def average_parent_snps(masked_matrix):
    num_samples, num_snps = masked_matrix.shape
    average_masked_matrix = np.zeros((int(num_samples / 2), num_snps))
    for i in range(int(num_samples / 2)):
        average_masked_matrix[i,:] = np.nanmean(masked_matrix[2*i:(2*i + 2),:], axis=0)
    return average_masked_matrix

def remove_AB_indIDs(ind_IDs):
    new_ind_IDs = []
    for i in range(int(len(ind_IDs)/2)):
        new_ind_IDs.append(ind_IDs[2*i][:-2])
    new_ind_IDs = np.array(new_ind_IDs)
    return new_ind_IDs

def add_AB_indIDs(ind_IDs):
    new_ind_IDs = []
    for i in range(len(ind_IDs)):
        new_ind_IDs.append(str(ind_IDs[i]) + '_A')
        new_ind_IDs.append(str(ind_IDs[i]) + '_B')
    new_ind_IDs = np.array(new_ind_IDs)
    return new_ind_IDs

def get_masked_matrix(beagle_filename, vcf_filename, beagle_or_vcf, is_masked, vit_filename, fbk_filename, tsv_filename, vit_or_fbk_or_tsv, fb_or_msp, num_ancestries, ancestry, average_parents, prob_thresh):
    if beagle_or_vcf == 1:
        gt_matrix, ind_IDs, rs_IDs = process_beagle(beagle_filename)
    elif beagle_or_vcf == 2:
        gt_matrix, ind_IDs, rs_IDs, positions = process_vcf(vcf_filename)
    else:
        sys.exit("Illegal value for beagle_or_vcf. Choose 1 for beagle file or 2 for vcf file.")
    if is_masked:
        if vit_or_fbk_or_tsv == 1:
            ancestry_matrix = process_vit(vit_filename)
        elif vit_or_fbk_or_tsv == 2:
            ancestry_matrix = process_fbk(fbk_filename, num_ancestries, prob_thresh)
        elif vit_or_fbk_or_tsv == 3:
            if fb_or_msp == 1:
                ancestry_matrix, gt_matrix = process_tsv_fb(tsv_filename, num_ancestries, prob_thresh, positions, gt_matrix)
            elif fb_or_msp == 2:
                ancestry_matrix, gt_matrix = process_tsv_msp(tsv_filename, positions, gt_matrix)
            else:
                sys.exit("Illegal value for fb_or_msp. Choose 1 for fb.tsv file or 2 for msp.tsv file.")
        else:
            sys.exit("Illegal value for vit_or_fbk_or_tsv. Choose 1 for vit file or 2 for fbk file or 3 for tsv file.")
        if vit_or_fbk_or_tsv == 1 or vit_or_fbk_or_tsv == 2:
            unique_ancestries = [str(i) for i in np.arange(1, num_ancestries+1)]
        else:
            unique_ancestries = [str(i) for i in np.arange(0, num_ancestries)]
        masked_matrices = mask(ancestry_matrix, gt_matrix, unique_ancestries)
        masked_matrix = masked_matrices[str(ancestry)].T
    else:
        masked_matrix = gt_matrix.T
    if average_parents:
        masked_matrix = average_parent_snps(masked_matrix)
        ind_IDs = remove_AB_indIDs(ind_IDs)    
    return masked_matrix, ind_IDs, rs_IDs

def process_labels_weights(labels_file, masked_matrix, ind_IDs, average_parents, is_weighted, save_masked_matrix, masked_matrix_filename):
    labels_df = pd.read_csv(labels_file, sep='\t')
    if average_parents:
        labels = np.array(labels_df['label'][labels_df['indID'].isin(ind_IDs)])
        label_ind_IDs = np.array(labels_df['indID'][labels_df['indID'].isin(ind_IDs)])
    else:
        temp_ind_IDs = remove_AB_indIDs(ind_IDs)
        labels = np.array(labels_df['label'][labels_df['indID'].isin(temp_ind_IDs)])
        labels = np.repeat(labels, 2)
        label_ind_IDs = np.array(labels_df['indID'][labels_df['indID'].isin(temp_ind_IDs)])
        label_ind_IDs = add_AB_indIDs(label_ind_IDs)
    keep_indices = [ind_IDs.tolist().index(x) for x in label_ind_IDs]
    ind_IDs = ind_IDs[keep_indices]
    masked_matrix = masked_matrix[keep_indices]
    if not is_weighted:
        weights = np.ones(len(labels))
    else:
        if average_parents:
            weights = np.array(labels_df['weight'][labels_df['indID'].isin(ind_IDs)])
        else:
            temp_ind_IDs = remove_AB_indIDs(ind_IDs)
            weights = np.array(labels_df['weight'][labels_df['indID'].isin(temp_ind_IDs)])
            weights = np.repeat(weights, 2)
        non_combined_indices = np.where(weights > 0)
        masked_matrix_new = masked_matrix[non_combined_indices]
        ind_IDs_new = ind_IDs[non_combined_indices]
        labels_new = labels[non_combined_indices]
        weights_new = weights[non_combined_indices]
        num_groups = - min(weights)
        if num_groups > 0:
            for i in range(1, num_groups+1):
                weight = -i
                combined_indices = np.where(weights == weight)
                combined_row = [np.nanmean(masked_matrix[combined_indices], axis=0)]
                masked_matrix_new = np.append(masked_matrix_new, combined_row, axis=0)
                ind_IDs_new = np.append(ind_IDs_new, 'combined_ind_' + str(i))
                labels_new = np.append(labels_new, labels[combined_indices[0][0]])
                weights_new = np.append(weights_new, 1)
        masked_matrix = masked_matrix_new
        ind_IDs = ind_IDs_new
        labels = labels_new
        weights = weights_new
    if save_masked_matrix:
        np.save(masked_matrix_filename, masked_matrix)
    return masked_matrix, ind_IDs, labels, weights

def center_masked_matrix(masked_matrix):
    masked_matrix -= np.nanmean(masked_matrix, axis=0)
    return masked_matrix