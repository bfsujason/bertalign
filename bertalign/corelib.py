import torch
import faiss
import numpy as np
import numba as nb
from sys import platform

def second_back_track(i, j, pointers, search_path, a_types):
    alignment = []
    while ( 1 ):
        j_offset = j - search_path[i][0]
        a = pointers[i][j_offset]
        s = a_types[a][0]
        t = a_types[a][1]
        src_range = [i - offset - 1 for offset in range(s)][::-1]
        tgt_range = [j - offset - 1 for offset in range(t)][::-1]
        alignment.append((src_range, tgt_range))

        i = i-s
        j = j-t
    
        if i == 0 and j == 0:
            return alignment[::-1]

@nb.jit(nopython=True, fastmath=True, cache=True)
def second_pass_align(src_vecs,
                      tgt_vecs,
                      src_lens,
                      tgt_lens,
                      w,
                      search_path,
                      align_types,
                      char_ratio,
                      skip,
                      margin=False,
                      len_penalty=False):
    """
    Perform the second-pass alignment to extract m-n bitext segments.
    Args:
        src_vecs: numpy array of shape (max_align-1, num_src_sents, embedding_size).
        tgt_vecs: numpy array of shape (max_align-1, num_tgt_sents, embedding_size).
        src_lens: numpy array of shape (max_align-1, num_src_sents).
        tgt_lens: numpy array of shape (max_align-1, num_tgt_sents).
        w: int. Predefined window size for the second-pass alignment.
        search_path: numpy array. Second-pass alignment search path.
        align_types: numpy array. Second-pass alignment types.
        char_ratio: float. Source to target length ratio.
        skip: float. Cost for instertion and deletion.
        margin: boolean. True if choosing modified cosine similarity score.
    Returns:
        pointers: numpy array recording best alignments for each DP cell.
    """
    # Intialize cost and backpointer matrix
    src_len = src_vecs.shape[1]
    tgt_len = tgt_vecs.shape[1]
    cost = np.zeros((src_len + 1, w), dtype=nb.float32)
    pointers = np.zeros((src_len + 1, w), dtype=nb.uint8)
  
    for i in range(src_len + 1):
        i_start = search_path[i][0]
        i_end = search_path[i][1]
        for j in range(i_start, i_end + 1):
            if i + j == 0:
                continue
            best_score = -np.inf
            best_a = -1
            for a in range(align_types.shape[0]):
                a_1 = align_types[a][0]
                a_2 = align_types[a][1]
                prev_i = i - a_1
                prev_j = j - a_2

                if prev_i < 0 or prev_j < 0 :  # no previous cell in DP table 
                    continue
                prev_i_start = search_path[prev_i][0]
                prev_i_end =  search_path[prev_i][1]
                if prev_j < prev_i_start or prev_j > prev_i_end: # out of bound of cost matrix
                    continue
                prev_j_offset = prev_j - prev_i_start
                score = cost[prev_i][prev_j_offset]

                if a_1 == 0 or a_2 == 0:  # deletion or insertion
                    cur_score = skip
                else:
                    cur_score = calculate_similarity_score(src_vecs,
                                                           tgt_vecs,
                                                           i, j, a_1, a_2, 
                                                           src_len, tgt_len,
                                                           margin=margin)
                    if len_penalty:
                        penalty = calculate_length_penalty(src_lens, tgt_lens, i, j,
                                                           a_1, a_2, char_ratio)
                        cur_score *= penalty
        
                score += cur_score
                if score > best_score:
                    best_score = score
                    best_a = a
            
            # Update cell(i, j) with the best score
            # and rescord the trace history.
            j_offset = j - i_start
            cost[i][j_offset] = best_score
            pointers[i][j_offset] = best_a
      
    return pointers

@nb.jit(nopython=True, fastmath=True, cache=True)
def calculate_similarity_score(src_vecs,
                               tgt_vecs,
                               src_idx,
                               tgt_idx,
                               src_overlap,
                               tgt_overlap,
                               src_len,
                               tgt_len,
                               margin=False):
  
    """
    Calulate the semantics-based similarity score of bitext segment.
    """
    src_v = src_vecs[src_overlap - 1, src_idx - 1, :]
    tgt_v = tgt_vecs[tgt_overlap - 1, tgt_idx - 1, :]
    similarity = nb_dot(src_v, tgt_v)
    if margin:
        tgt_neighbor_ave_sim = calculate_neighbor_similarity(src_v, 
                                                             tgt_overlap,
                                                             tgt_idx,
                                                             tgt_len,
                                                             tgt_vecs)
    
        src_neighbor_ave_sim = calculate_neighbor_similarity(tgt_v,
                                                             src_overlap,
                                                             src_idx,
                                                             src_len,
                                                             src_vecs)
    
        neighbor_ave_sim = (tgt_neighbor_ave_sim + src_neighbor_ave_sim) / 2
        similarity -= neighbor_ave_sim

    return similarity

@nb.jit(nopython=True, fastmath=True, cache=True)
def calculate_neighbor_similarity(vec, overlap, sent_idx, sent_len, db):
    left_idx = sent_idx - overlap
    right_idx = sent_idx + 1
    
    if right_idx <= sent_len:
        right_embed = db[0, right_idx - 1, :]
        neighbor_right_sim = nb_dot(vec, right_embed)
    else:
        neighbor_right_sim = 0
 
    if left_idx > 0:
        left_embed = db[0, left_idx - 1, :]
        neighbor_left_sim = nb_dot(vec, left_embed)
    else:
        neighbor_left_sim = 0
    
    neighbor_ave_sim = neighbor_left_sim + neighbor_right_sim
    if neighbor_right_sim and neighbor_left_sim:
        neighbor_ave_sim /= 2
    
    return neighbor_ave_sim

@nb.jit(nopython=True, fastmath=True, cache=True)
def calculate_length_penalty(src_lens,
                             tgt_lens,
                             src_idx,
                             tgt_idx,
                             src_overlap,
                             tgt_overlap,
                             char_ratio):
    """
    Calculate the length-based similarity score of bitext segment.
    Args:
        src_lens: numpy array. Source sentence lengths vector.
        tgt_lens: numpy array. Target sentence lengths vector.
        src_idx: int. Source sentence index.
        tgt_idx: int. Target sentence index.
        src_overlap: int. Number of sentences in source segment.
        tgt_overlap: int. Number of sentences in target segment.
        char_ratio: float. Source to target sentence length ratio.
    Returns:
        length_penalty: float. Similarity score based on length differences.
    """
    src_l = src_lens[src_overlap - 1, src_idx - 1]
    tgt_l = tgt_lens[tgt_overlap - 1, tgt_idx - 1]
    tgt_l = tgt_l * char_ratio
    min_len = min(src_l, tgt_l)
    max_len = max(src_l, tgt_l)
    length_penalty = np.log2(1 + min_len / max_len)
    return length_penalty

@nb.jit(nopython=True, fastmath=True, cache=True)
def nb_dot(x, y):
    return np.dot(x,y)

def find_second_search_path(align, w, src_len, tgt_len):
    """
    Convert 1-1 first-pass alignment to the second-round path.
    The indices along X-axis and Y-axis must be consecutive.
    Args:
        align: list of tuples. First-pass alignment results.
        w: int. Predefined window size for the second path.
        src_len: int. Number of source sentences.
        tgt_len: int. Number of target sentences.
    Returns:
        path: numpy array. Search path for the second-pass alignment.
    """
    # Ajust the first-alignment result
    # so that the last bead is (src_len, tgt_len).
    last_bead_src = align[-1][0]
    last_bead_tgt = align[-1][1]
    if last_bead_src != src_len:
        if last_bead_tgt == tgt_len:
            align.pop()
        align.append((src_len, tgt_len))
    else:
        if last_bead_tgt != tgt_len:
            align.pop()
            align.append((src_len, tgt_len))
    
    """
    Find the search path for each row.
    """
    prev_src, prev_tgt = 0, 0
    path = []
    max_w = -np.inf
    for src, tgt in align:
        # Limit the search path in a rectangle with the width
        # along the Y axis being (upper_bound - lower_bound).
        lower_bound = max(0, prev_tgt - w)
        upper_bound = min(tgt_len, tgt + w)
        path.extend([(lower_bound, upper_bound) for id in range(prev_src+1, src+1)])
        prev_src, prev_tgt = src, tgt
        width = upper_bound - lower_bound
        if width > max_w:
            max_w = width
    path = [path[0]] + path # add the search path for row 0
    return max_w + 1, np.array(path)

def first_back_track(i, j, pointers, search_path, a_types):
    """
    Retrieve 1-1 alignments from the first-pass DP table.
    Args:
        i: int. Number of source sentences.
        j: int. Number of target sentences.
        pointers: numpy array. Backpointer matrix of first-pass alignment.
        search_path: numpy array. First-pass search path.
        a_types: numpy array. First-pass alignment types.
    Returns:
        alignment: list of tuples for 1-1 alignments.
    """
    alignment = []
    while ( 1 ):
        j_offset = j - search_path[i][0]
        a = pointers[i][j_offset]
        s = a_types[a][0]
        t = a_types[a][1]
        if a == 2: # best 1-1 alignment
            alignment.append((i, j))

        i = i-s
        j = j-t
    
        if i == 0 and j == 0: # if reaching the origin
            return alignment[::-1]

@nb.jit(nopython=True, fastmath=True, cache=True)
def first_pass_align(src_len,
                     tgt_len,
                     w,
                     search_path,
                     align_types,
                     dist,
                     index
                     ):
    """
    Perform the first-pass alignment to extract only 1-1 bitext segments.
    Args:
        src_len: int. Number of source sentences.
        tgt_len: int. Number of target sentences.
        w: int. Window size for the first-pass alignment.
        search_path: numpy array. Search path for the first-pass alignment.
        align_types: numpy array. Alignment types for the first-pass alignment.
        dist: numpy array. Distance matrix for top-k similar vecs.
        index: numpy array. Index matrix for top-k similar vecs.
    Returns:
        pointers: numpy array recording best alignments for each DP cell.
    """
    # Initialize cost and backpointer matrix.
    cost = np.zeros((src_len + 1, 2 * w + 1), dtype=nb.float32)
    pointers = np.zeros((src_len + 1, 2 * w + 1), dtype=nb.uint8)
  
    top_k = index.shape[1]

    for i in range(src_len + 1):
        i_start = search_path[i][0]
        i_end = search_path[i][1]
        for j in range(i_start, i_end + 1):
            if i + j == 0: # initialize the origin with zero
                continue
            best_score = -np.inf
            best_a = -1
            for a in range(align_types.shape[0]):
                a_1 = align_types[a][0]
                a_2 = align_types[a][1]
                prev_i = i - a_1
                prev_j = j - a_2
                if prev_i < 0 or prev_j < 0 :  # no previous cell 
                    continue
                prev_i_start = search_path[prev_i][0]
                prev_i_end =  search_path[prev_i][1]
                if prev_j < prev_i_start or prev_j > prev_i_end: # out of bound of cost matrix
                    continue
                prev_j_offset = prev_j - prev_i_start
                score = cost[prev_i][prev_j_offset]
                
                # Extract the score for 1-1 bead from faiss.
                if a_1 > 0 and a_2 > 0:
                    for k in range(top_k):
                        if index[i-1][k] == j - 1:
                            score += dist[i-1][k]
                if score > best_score:
                    best_score = score
                    best_a = a
            
            # Update cell(i, j) with the best score
            # and rescord the trace history.
            j_offset = j - i_start
            cost[i][j_offset] = best_score
            pointers[i][j_offset] = best_a

    return pointers

def find_first_search_path(src_len,
                           tgt_len,
                           min_win_size = 250,
                           percent=0.06):
    """
    Find the window size and search path for the first-pass alignment.
    Args:
        src_len: int. Number of source sentences.
        tgt_len: int. Number of target sentences.
        min_win_size: int. Minimum window size.
        percent. float. Percent of longer sentences.
    Returns:
        win_size: int. Window size along the diagonal of the DP table.
        search_path: numpy array of shape (src_len + 1, 2), containing the start
                     and end index of target sentences for each source sentence.
                     One extra row is added in the search_path for the calculation
                     of deletions and omissions.
    """
    win_size = max(min_win_size, int(max(src_len, tgt_len) * percent))
    search_path = []
    yx_ratio = tgt_len / src_len
    for i in range(0, src_len + 1):
        center = int(yx_ratio * i)
        win_start = max(0, center - win_size)
        win_end = min(center + win_size, tgt_len)
        search_path.append([win_start, win_end])
    return win_size, np.array(search_path)

def get_alignment_types(max_alignment_size):
    """
    Get all the possible alignment types.
    Args:
        max_alignment_size: int. Source sentence number +
                                 Target sentence number <= this value.
    Returns:
        alignment_types: numpy array.
    """
    alignment_types = [[0,1], [1,0]]
    for x in range(1, max_alignment_size):
        for y in range(1, max_alignment_size):
            if x + y <= max_alignment_size:
                alignment_types.append([x, y])    
    return np.array(alignment_types)

def find_top_k_sents(src_vecs, tgt_vecs, k=3):
    """
    Find the top_k similar vecs in tgt_vecs for each vec in src_vecs.
    Args:
        src_vecs: numpy array of shape (num_src_sents, embedding_size).
        tgt_vecs: numpy array of shape (num_tgt_sents, embedding_size).
        k: int. Number of most similar target sentences.
    Returns:
        D: numpy array. Similarity score matrix of shape (num_src_sents, k).
        I: numpy array. Target index matrix of shape (num_src_sents, k).
    """
    embedding_size = src_vecs.shape[1]
    if torch.cuda.is_available() and platform == 'linux': # GPU version
        res = faiss.StandardGpuResources() 
        index = faiss.IndexFlatIP(embedding_size)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(tgt_vecs) 
        D, I = gpu_index.search(src_vecs, k)
    else: # CPU version
        index = faiss.IndexFlatIP(embedding_size)
        index.add(tgt_vecs)
        D, I = index.search(src_vecs, k)
    return D, I
