import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def rkhs_mmd(samples_1, samples_2, bandwidth):  # two given sample groups, shape (N*dim)
	# m, dim = np.shape(samples_1)
	# n, _ = np.shape(samples_2)
	#
	# def rbf_kernel(z_1, z_2, bandwidth):
	# 	z_1_expand = np.expand_dims(z_1, axis=1)
	# 	dist_square = np.sum((z_1_expand - z_2)**2, axis=-1)
	# 	kernel_matrix = np.exp(-dist_square/bandwidth)
	# 	return kernel_matrix
	#
	# kxx = rbf_kernel(samples_1, samples_1, bandwidth)
	# kyy = rbf_kernel(samples_2, samples_2, bandwidth)
	# kxy = rbf_kernel(samples_1, samples_2, bandwidth)
	# hxy = kxx + kyy - 2*kxy
	#
	# return np.mean(hxy)
	# gamma = bandwidth
	gamma = 1/bandwidth
	kxx = rbf_kernel(samples_1, samples_1, gamma)
	kyy = rbf_kernel(samples_2, samples_2, gamma)
	kxy = rbf_kernel(samples_1, samples_2, gamma)
	return kxx.mean() + kyy.mean() - 2 * kxy.mean()



def calculate_mmd(sequence_1, sequence_2, bandwidth, mode='MMDA'):  # compute the mmd between sequences, shape (N*len*dim)
    sequence_1 = sequence_1.transpose(0,2,1)
    sequence_2 = sequence_2.transpose(0,2,1)
    _, seq_len, dim = np.shape(sequence_1)
    result = 0.
    if mode == 'MMDA':
        for frames in range(seq_len):
            result += rkhs_mmd(sequence_1[:, frames, :], sequence_2[:, frames, :], bandwidth)/seq_len
    elif mode == 'MMDS':
        flat_seq_1 = np.reshape(sequence_1, (-1, dim*seq_len))
        flat_seq_2 = np.reshape(sequence_2, (-1, dim*seq_len))
        result = rkhs_mmd(flat_seq_1, flat_seq_2, bandwidth)
    else:
        raise Exception('undefined mode')
    return result


def mmd_function(gen, real, mode='MMDA'):
	# labels = np.unique(y)
	new_r = 0
	result_list = []
	for j in range(-7,10):
		new_new_r = calculate_mmd(gen, real, 10 ** j, mode=mode)
		if new_new_r > new_r:
			new_r = new_new_r
		# result_list.append(new_new_r)
		result_list.append(new_r) # original
	result = np.mean(result_list) # original
	# result = np.max(result_list)
	return result

