import math

import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity


# Helper for similarity-based metrics
def _get_similarity_scores(
    retrieved_embeddings: list[np.ndarray], ground_truth_embeddings: list[np.ndarray]
) -> np.ndarray:
    """Calculates a similarity matrix between retrieved and ground truth embeddings."""
    if not retrieved_embeddings or not ground_truth_embeddings:
        return np.array([[]])
    # Ensure embeddings are 2D arrays for cosine_similarity
    ret_emb_np = np.array(retrieved_embeddings)
    gt_emb_np = np.array(ground_truth_embeddings)
    if ret_emb_np.ndim == 1:  # Single retrieved embedding
        ret_emb_np = ret_emb_np.reshape(1, -1)
    if gt_emb_np.ndim == 1:  # Single ground truth embedding
        gt_emb_np = gt_emb_np.reshape(1, -1)
    return cosine_similarity(ret_emb_np, gt_emb_np)


def precision_at_k_exact(
    retrieved_chunks: list[str], ground_truth_chunks: list[str], k: int
) -> float:
    """Calculates Precision@k based on exact string matches.

    Precision@k measures the fraction of relevant items among the first `k`
    retrieved items. Relevance is determined by an exact match between a
    retrieved chunk and any ground truth chunk.

    Args:
        retrieved_chunks (list[str]): The list of strings retrieved by the system.
            Assumes that the retrieved chunks are unique.
        ground_truth_chunks (list[str]): The list of ground truth strings
            considered relevant. Assumes that the ground truth chunks are unique.
        k (int): The number of top retrieved chunks to evaluate.

    Returns:
        float: The Precision@k score, a value between 0.0 and 1.0.
            Returns 0.0 if `k` is 0 or `retrieved_chunks` is empty.

    Raises:
        ValueError: If `ground_truth_chunks` is empty.
    """
    if not retrieved_chunks or k == 0:
        return 0.0
    if not ground_truth_chunks:
        raise ValueError("Ground truth chunks cannot be empty")

    top_k_retrieved = retrieved_chunks[:k]
    relevant_hits = 0
    # Use set intersection to find relevant hits, assuming ground truth chunks and top k retrieved chunks are unique
    relevant_hits = len(set(top_k_retrieved) & set(ground_truth_chunks))

    return relevant_hits / k


def precision_at_k_similarity(
    retrieved_embeddings: list[np.ndarray],
    ground_truth_embeddings: list[np.ndarray],
    k: int,
    similarity_threshold: float = 0.8,
) -> float:
    """Calculates Precision@k based on cosine similarity between embeddings.

    Precision@k measures the fraction of relevant items among the top `k`
    retrieved items. A retrieved item is considered relevant if its cosine
    similarity to *any* ground truth item's embedding is at or above the
    `similarity_threshold`.

    Args:
        retrieved_embeddings (list[np.ndarray]): list of embedding vectors
            (NumPy arrays) for the retrieved items.
        ground_truth_embeddings (list[np.ndarray]): list of embedding vectors
            (NumPy arrays) for the ground truth items.
        k (int): The number of top retrieved embeddings to consider.
        similarity_threshold (float, optional): The cosine similarity threshold
            for relevance. Defaults to 0.8.

    Returns:
        float: The Precision@k score (a value between 0.0 and 1.0).
            Returns 0.0 if `k` is 0 or `retrieved_embeddings` is empty.

    Raises:
        ValueError: If `ground_truth_embeddings` is empty.
    """
    if not retrieved_embeddings or k == 0:
        return 0.0
    if not ground_truth_embeddings:
        raise ValueError("Ground truth embeddings cannot be empty")

    top_k_retrieved_embeddings = retrieved_embeddings[:k]

    similarity_matrix = _get_similarity_scores(
        top_k_retrieved_embeddings, ground_truth_embeddings
    )

    if similarity_matrix.size == 0:
        raise ValueError(
            "Similarity matrix cannot be empty",
            "This code should not be reached, please check the inputs",
        )

    relevant_hits = 0
    # For each of the top k retrieved items, check if it's similar to ANY ground truth item.
    for i in range(similarity_matrix.shape[0]):  # Iterate over retrieved items
        if np.any(similarity_matrix[i, :] >= similarity_threshold):
            relevant_hits += 1

    return relevant_hits / k


def recall_at_k_exact(
    retrieved_chunks: list[str], ground_truth_chunks: list[str], k: int
) -> float:
    """
    Calculates Recall@k based on exact string matches.

    Recall@k measures the fraction of ground truth items that appear in the top `k`
    retrieved items. A ground truth item is considered recalled if it exactly matches
    any of the top `k` retrieved items.

    Args:
        retrieved_chunks (list[str]): list of strings representing the retrieved items.
        ground_truth_chunks (list[str]): list of strings representing the ground truth items.
        k (int): The number of top retrieved chunks to consider.

    Returns:
        float: The Recall@k score (a value between 0.0 and 1.0).
            Returns 0.0 if `k` is 0 or `retrieved_chunks` is empty.

    Raises:
        ValueError: If `ground_truth_chunks` is empty.
    """
    if not retrieved_chunks or k == 0:
        return 0.0
    if not ground_truth_chunks:
        raise ValueError("Ground truth chunks cannot be empty")

    top_k_retrieved = retrieved_chunks[:k]
    relevant_hits = 0
    # Use a set for faster lookups of ground truth for uniqueness
    ground_truth_set = set(ground_truth_chunks)
    # Count relevant hits (ground truth items found in top_k retrieved items)
    retrieved_set = set(top_k_retrieved)
    hit_gt_items = ground_truth_set.intersection(retrieved_set)
    relevant_hits = len(hit_gt_items)

    return relevant_hits / len(ground_truth_set)


def recall_at_k_similarity(
    retrieved_embeddings: list[np.ndarray],
    ground_truth_embeddings: list[np.ndarray],
    k: int,
    similarity_threshold: float = 0.8,
) -> float:
    """
    Calculates Recall@k based on cosine similarity between embedding vectors.

    Recall@k measures the fraction of ground truth items that are similar to any of the top `k`
    retrieved items. A ground truth item is considered recalled if its cosine similarity
    with any of the top `k` retrieved items exceeds the specified threshold.

    Args:
        retrieved_embeddings (list[np.ndarray]): list of embedding vectors representing the retrieved items.
        ground_truth_embeddings (list[np.ndarray]): list of embedding vectors representing the ground truth items.
        k (int): The number of top retrieved embeddings to consider.
        similarity_threshold (float, optional): The minimum cosine similarity threshold for considering
            a ground truth item as recalled. Defaults to 0.8.

    Returns:
        float: The Recall@k score (a value between 0.0 and 1.0).
            Returns 0.0 if `k` is 0 or `retrieved_embeddings` is empty.

    Raises:
        ValueError: If `ground_truth_embeddings` is empty or if the similarity matrix computation fails.
    """
    if not ground_truth_embeddings:
        raise ValueError("Ground truth embeddings cannot be empty")
    if not retrieved_embeddings or k == 0:
        return 0.0

    top_k_retrieved_embeddings = retrieved_embeddings[:k]

    similarity_matrix = _get_similarity_scores(
        top_k_retrieved_embeddings, ground_truth_embeddings
    )

    if similarity_matrix.size == 0:
        raise ValueError(
            "Similarity matrix cannot be empty",
            "This code should not be reached, please check the inputs",
        )

    # For each ground truth item, check if it's similar to ANY of the top k retrieved items.
    # This counts how many ground truth items were successfully recalled.
    recalled_gt_count = 0
    for j in range(similarity_matrix.shape[1]):  # Iterate over ground truth items
        if np.any(similarity_matrix[:, j] >= similarity_threshold):
            recalled_gt_count += 1

    return recalled_gt_count / len(ground_truth_embeddings)


def f1_at_k_exact(
    retrieved_chunks: list[str], ground_truth_chunks: list[str], k: int
) -> float:
    """Calculates F1-score@k based on exact string matches.

    F1-score@k is the harmonic mean of Precision@k and Recall@k, providing a balanced
    measure of retrieval performance that considers both precision and recall.

    Args:
        retrieved_chunks (list[str]): The list of strings retrieved by the system.
            Assumes that the retrieved chunks are unique.
        ground_truth_chunks (list[str]): The list of ground truth strings
            considered relevant. Assumes that the ground truth chunks are unique.
        k (int): The number of top retrieved chunks to evaluate.

    Returns:
        float: The F1-score@k value, between 0.0 and 1.0.
            Returns 0.0 if either precision or recall is 0.
    """
    precision = precision_at_k_exact(retrieved_chunks, ground_truth_chunks, k)
    recall = recall_at_k_exact(retrieved_chunks, ground_truth_chunks, k)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def f1_at_k_similarity(
    retrieved_embeddings: list[np.ndarray],
    ground_truth_embeddings: list[np.ndarray],
    k: int,
    similarity_threshold: float = 0.8,
) -> float:
    """Calculates F1-score@k based on cosine similarity between embeddings.

    F1-score@k is the harmonic mean of Precision@k and Recall@k, providing a balanced
    measure of retrieval performance that considers both precision and recall.
    Relevance is determined by cosine similarity exceeding the specified threshold.

    Args:
        retrieved_embeddings (list[np.ndarray]): The list of embedding vectors retrieved by the system.
        ground_truth_embeddings (list[np.ndarray]): The list of ground truth embedding vectors
            considered relevant.
        k (int): The number of top retrieved embeddings to evaluate.
        similarity_threshold (float, optional): The minimum cosine similarity threshold
            for considering two embeddings as similar. Defaults to 0.8.

    Returns:
        float: The F1-score@k value, between 0.0 and 1.0.
            Returns 0.0 if either precision or recall is 0.
    """
    precision = precision_at_k_similarity(
        retrieved_embeddings, ground_truth_embeddings, k, similarity_threshold
    )
    recall = recall_at_k_similarity(
        retrieved_embeddings, ground_truth_embeddings, k, similarity_threshold
    )

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def log_rank_score(rank, n, gamma=1.0):
    """Compute log-rank score for a given 1-based rank."""
    if n == 1:
        # If there's only one item, its rank must be 1 for it to be considered.
        # The score reflects a perfect ranking for this single item if it's relevant.
        return 1.0

    # For n > 1:
    # Note: If gamma = 0, this formula results in math.log(1)/math.log(1) = 0/0 = NaN.
    # Users should ensure gamma > 0 or handle the gamma=0 case separately if needed,
    # as the behavior for gamma=0 is not explicitly defined by this formula here.
    numerator = math.log(1 + gamma * (rank - 1))
    denominator = math.log(1 + gamma * (n - 1))

    # With n > 1 and assuming gamma >= 0:
    # If gamma > 0, denominator = log(1 + positive) > log(1) = 0. So, no division by zero.
    # If gamma = 0, denominator = log(1) = 0. Numerator is also log(1)=0. This leads to NaN.
    # The function relies on the caller to provide meaningful gamma values.
    return 1 - (numerator / denominator)


def hybrid_log_rank_score_exact(
    retrieved_chunks: list[str],
    ground_truth_chunks: list[str],
    gamma: float = 1.0,
    alpha: float = 0.5,
) -> float:
    """
    Compute hybrid score that combines recall and rank-based scoring using exact matches.

    Parameters:
    - retrieved_chunks: list of chunk IDs (ranked top to bottom)
    - ground_truth_chunks: list of relevant chunk IDs (must not be empty).
    - gamma: log curve control parameter for log_rank_score.
    - alpha: recall vs rank-quality tradeoff [0 (only rank), 1 (only recall)].

    Returns:
    - hybrid_score: float between 0 and 1.

    Raises:
        ValueError: If `ground_truth_chunks` is empty.
    """
    if not ground_truth_chunks:
        raise ValueError(
            "Ground truth chunks cannot be empty for hybrid_log_rank_score_exact."
        )

    N = len(retrieved_chunks)
    if N == 0:  # No retrieved chunks
        return 0.0  # Recall is 0, rank quality is 0

    gt_set = set(ground_truth_chunks)  # Not empty due to the check above
    retrieved_set = set(retrieved_chunks)

    retrieved_relevant = gt_set.intersection(retrieved_set)

    # Recall component
    recall = len(retrieved_relevant) / len(gt_set)  # len(gt_set) > 0

    # Rank-quality component
    rank_scores = []
    # N (len(retrieved_chunks)) is > 0 here.
    for chunk in retrieved_relevant:
        try:
            rank = retrieved_chunks.index(chunk) + 1  # 1-based rank
            score = log_rank_score(rank, N, gamma)
            rank_scores.append(score)
        except ValueError:
            # This should ideally not happen if chunk is from retrieved_relevant.
            # Handles cases like duplicates in retrieved_chunks not reflected in retrieved_set logic if any.
            pass

    rank_quality = sum(rank_scores) / len(gt_set)  # len(gt_set) > 0

    # Hybrid score
    return alpha * recall + (1 - alpha) * rank_quality


def hybrid_log_rank_score_similarity(
    retrieved_embeddings: list[np.ndarray],
    ground_truth_embeddings: list[np.ndarray],
    similarity_threshold: float = 0.8,
    gamma: float = 1.0,
    alpha: float = 0.5,
) -> float:
    """
    Compute hybrid score that combines recall and rank-based scoring using cosine similarity.

    Parameters:
    - retrieved_embeddings: list of embedding vectors for retrieved items.
    - ground_truth_embeddings: list of embedding vectors for ground truth items (must not be empty).
    - similarity_threshold: The cosine similarity threshold for relevance.
    - gamma: log curve control parameter for log_rank_score.
    - alpha: recall vs rank-quality tradeoff [0 (only rank), 1 (only recall)].

    Returns:
    - hybrid_score: float between 0 and 1.

    Raises:
        ValueError: If `ground_truth_embeddings` is empty or if similarity matrix calculation fails unexpectedly.
    """
    if not ground_truth_embeddings:
        raise ValueError(
            "Ground truth embeddings cannot be empty for hybrid_log_rank_score_similarity."
        )

    num_retrieved = len(retrieved_embeddings)
    num_gt = len(ground_truth_embeddings)  # Known to be > 0

    if num_retrieved == 0:
        return 0.0  # No retrieved items, so recall is 0, rank quality is 0.

    similarity_matrix = _get_similarity_scores(
        retrieved_embeddings, ground_truth_embeddings
    )

    # _get_similarity_scores handles empty retrieved_embeddings internally.
    # If num_retrieved > 0 and num_gt > 0, similarity_matrix should be valid.
    if similarity_matrix.shape != (num_retrieved, num_gt):
        # This might indicate an issue if _get_similarity_scores behaves unexpectedly with non-empty inputs
        # or if inputs to _get_similarity_scores were altered (e.g. by slicing) before call in other contexts.
        # For this function, direct pass-through should yield expected dimensions.
        pass  # Allow to proceed, subsequent operations might fail if shape is truly problematic.

    # Recall component: Fraction of ground truth items similar to at least one retrieved item.
    recalled_gt_indices = set()
    # similarity_matrix shape is (num_retrieved, num_gt)
    for j in range(num_gt):  # Iterate over ground truth items (columns)
        if np.any(similarity_matrix[:, j] >= similarity_threshold):
            recalled_gt_indices.add(j)

    recall = len(recalled_gt_indices) / num_gt

    # Rank-quality component:
    # For each *recalled* ground truth item, find the log_rank_score of the *best-ranked* retrieved item that hit it.
    # Sum these scores and normalize by the total number of ground truth items.

    best_rank_for_recalled_gt = {}  # Stores {gt_index: best_rank (1-based)}
    for gt_idx in recalled_gt_indices:
        min_rank_for_this_gt = float("inf")
        for i in range(num_retrieved):  # Iterate over retrieved items (rows)
            if similarity_matrix[i, gt_idx] >= similarity_threshold:
                min_rank_for_this_gt = min(min_rank_for_this_gt, i + 1)
        if min_rank_for_this_gt != float("inf"):
            best_rank_for_recalled_gt[gt_idx] = min_rank_for_this_gt

    rank_scores_sum = 0.0
    # N for log_rank_score is num_retrieved (which is > 0 here)
    for _, rank in best_rank_for_recalled_gt.items():
        rank_scores_sum += log_rank_score(rank, num_retrieved, gamma)

    rank_quality = rank_scores_sum / num_gt  # num_gt > 0

    return alpha * recall + (1 - alpha) * rank_quality


def bleu_score(retrieved_chunk: str, ground_truth_chunks: list[str]) -> float:
    """
    Calculates BLEU score for a single retrieved chunk against multiple ground truth references using sacrebleu.
    Args:
        retrieved_chunk: The retrieved string (hypothesis).
        ground_truth_chunks: A list of ground truth strings (references).
    Returns:
        BLEU score (0-100 scale from sacrebleu, will be divided by 100).
    """
    if not ground_truth_chunks:
        return 0.0  # No references to compare against
    if not retrieved_chunk:
        # If hypothesis is empty, BLEU is 0, unless all references are also empty.
        if all(not gt for gt in ground_truth_chunks):
            return 1.0  # Or 100.0 then scaled. Let's return 0.0 to be consistent with no overlap.
            # sacrebleu sentence_bleu with empty hypothesis and non-empty refs gives 0.
            # If refs are also empty, it might raise an error or give 0.
            # Let's ensure refs are not all empty for the 1.0 case
        is_any_ref_non_empty = any(bool(gt) for gt in ground_truth_chunks)
        if not is_any_ref_non_empty:  # All refs are empty, hypothesis is empty
            return 1.0
        return 0.0  # Empty hypothesis, at least one non-empty ref

    # sacrebleu.sentence_bleu expects a single hypothesis string and a list of reference strings.
    # It returns a BLEUScore object. The actual score is in score.score
    # Note: sacrebleu scores are typically 0-100. We should scale to 0-1 for consistency with other metrics.
    try:
        # Sacrebleu's sentence_bleu can take a list of strings for references
        bleu_result = sacrebleu.sentence_bleu(
            retrieved_chunk,
            ground_truth_chunks,  # tokenize="flores101"
        )  # Using a common tokenizer
        return bleu_result.score / 100.0
    except Exception as e:
        # Could be due to various issues, e.g., empty references with non-empty hypothesis
        # For robustness, return 0.0 on error. Check sacrebleu docs for specific error handling.
        # print(f"Sacrebleu error: {e}") # For debugging
        print(f"Sacrebleu error: {e}")
        return 0.0


def corpus_bleu_score(
    retrieved_chunks: list[str], ground_truth_chunks: list[str]
) -> float:
    """
    Calculates an aggregated BLEU score based on maximizing BLEU for each ground truth chunk.

    For each ground_truth_chunk:
    1. It finds the retrieved_chunk that maximizes the BLEU score when paired with
       the current ground_truth_chunk (the single ground_truth_chunk is used as the reference list for bleu_score).
    2. This maximum BLEU score (a float between 0.0 and 1.0) is recorded.
    The final corpus BLEU score is the average of these recorded maximum BLEU scores.

    This aggregation strategy is analogous to how `corpus_rouge_scores` aggregates ROUGE metrics.

    Args:
        retrieved_chunks (list[str]): A list of retrieved strings (hypotheses).
        ground_truth_chunks (list[str]): A list of ground truth strings (references).
            Each string in this list is treated as an individual reference to be maximized against.

    Returns:
        float: The aggregated BLEU score, a value between 0.0 and 1.0.
               Returns 0.0 if `retrieved_chunks` is empty and `ground_truth_chunks` is not.
               The behavior for both empty `retrieved_chunks` and empty `ground_truth_chunks`
               is governed by the `ValueError` for empty `ground_truth_chunks`.

    Raises:
        ValueError: If `ground_truth_chunks` is empty.
    """
    if not ground_truth_chunks:
        raise ValueError("ground_truth_chunks cannot be empty for corpus_bleu.")

    if not retrieved_chunks:
        # If there are no retrieved chunks, no matches can be found for any ground truth.
        # Since ground_truth_chunks is guaranteed to be non-empty here,
        # the average max BLEU score will be 0.
        return 0.0

    num_gt = len(ground_truth_chunks)
    sum_of_max_bleu_scores = 0.0

    for gt_chunk in ground_truth_chunks:
        max_bleu_for_this_gt = 0.0  # BLEU scores are between 0.0 and 1.0

        # Determine the best BLEU score for the current gt_chunk against all retrieved_chunks
        for r_chunk in retrieved_chunks:
            # bleu_score expects a list of references, so [gt_chunk] is used.
            # The bleu_score function handles cases like r_chunk and/or gt_chunk being empty.
            current_bleu = bleu_score(r_chunk, [gt_chunk])
            if current_bleu > max_bleu_for_this_gt:
                max_bleu_for_this_gt = current_bleu

        sum_of_max_bleu_scores += max_bleu_for_this_gt

    # The average of the maximum BLEU scores found for each ground truth chunk.
    # num_gt is guaranteed to be > 0 here due to the initial check.
    return sum_of_max_bleu_scores / num_gt


def rouge_score(
    retrieved_chunk: str, ground_truth_chunk: str
) -> dict[str, dict[str, float]]:
    """
    Calculates ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    Args:
        retrieved_chunk: The retrieved string (hypothesis).
        ground_truth_chunk: The ground truth string (reference).
    Returns:
        A dictionary where keys are ROUGE types ('rouge1', 'rouge2', 'rougeL')
        and values are dictionaries containing 'precision', 'recall', and 'fmeasure' scores.
    """
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    if not retrieved_chunk or not ground_truth_chunk:
        # If either is empty, ROUGE scores are typically 0 for P, R, F.
        return {
            rtype: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
            for rtype in rouge_types
        }

    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    # The scores object is a dict where keys are ROUGE types (e.g., 'rouge1')
    # and values are Score objects (precision, recall, fmeasure).
    scores_obj = scorer.score(ground_truth_chunk, retrieved_chunk)

    # Extract P, R, F scores for convenience
    output_scores = {}
    for rtype in rouge_types:
        output_scores[rtype] = {
            "precision": scores_obj[rtype].precision,
            "recall": scores_obj[rtype].recall,
            "fmeasure": scores_obj[rtype].fmeasure,
        }
    return output_scores


def corpus_rouge_scores(
    retrieved_chunks: list[str], ground_truth_chunks: list[str]
) -> dict[str, dict[str, float]]:
    """
    Calculates ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) using a specific aggregation.

    For each ROUGE type (e.g., ROUGE-1):
    1. For each ground_truth_chunk:
        a. It finds the retrieved_chunk that maximizes the 'recall' for that ROUGE type
           when paired with the current ground_truth_chunk.
        b. The 'precision', 'recall', and 'fmeasure' from this specific
           (retrieved_chunk_maximizing_recall, ground_truth_chunk) pair are recorded.
    2. The recorded 'precision' scores (one for each ground_truth_chunk) are averaged
       to get the final corpus 'precision' for that ROUGE type.
    3. The same averaging is done for 'recall' and 'fmeasure' scores.

    Args:
        retrieved_chunks (list[str]): A list of retrieved strings (hypotheses).
        ground_truth_chunks (list[str]): A list of ground truth strings (references).

    Returns:
        dict[str, dict[str, float]]: A dictionary where keys are ROUGE types
                                     ('rouge1', 'rouge2', 'rougeL') and values are
                                     dictionaries containing the aggregated 'precision',
                                     'recall', and 'fmeasure' scores.
                                     Returns all 0.0 if retrieved_chunks is empty.
    Raises:
        ValueError: If `ground_truth_chunks` is empty.
    """
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    default_metrics = {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}

    final_corpus_scores = {rtype: default_metrics.copy() for rtype in rouge_types}

    if not retrieved_chunks:
        return final_corpus_scores  # All zeros if no retrieved chunks

    if not ground_truth_chunks:
        raise ValueError("ground_truth_chunks cannot be empty for corpus_rouge_scores.")

    num_gt = len(ground_truth_chunks)

    # Stores the sum of P, R, F1 from the best-recall pairs for each ROUGE type
    summed_scores_for_corpus = {rtype: default_metrics.copy() for rtype in rouge_types}

    for gt_chunk in ground_truth_chunks:
        # For this gt_chunk, find the best r_chunk for each ROUGE type based on its recall
        # These will store the P,R,F dict of the best pair for each ROUGE type
        best_scores_for_this_gt_per_rtype = dict.fromkeys(rouge_types)
        # Tracks the max recall found for this gt_chunk for each ROUGE type
        max_recall_for_this_gt_per_rtype = dict.fromkeys(rouge_types, -1.0)

        for r_chunk in retrieved_chunks:
            # rouge_score returns: {'rouge1': {'precision':..,'recall':..,'fmeasure':..}, ...}
            pair_all_rouge_scores = rouge_score(r_chunk, gt_chunk)

            for rtype in rouge_types:
                current_rtype_metrics = pair_all_rouge_scores[rtype]
                current_recall = current_rtype_metrics["recall"]

                if current_recall > max_recall_for_this_gt_per_rtype[rtype]:
                    max_recall_for_this_gt_per_rtype[rtype] = current_recall
                    best_scores_for_this_gt_per_rtype[rtype] = current_rtype_metrics

        # After checking all r_chunks for the current gt_chunk:
        # Add the P, R, F1 from the "best recall" pair to the overall sums
        for rtype in rouge_types:
            scores_from_best_pair = best_scores_for_this_gt_per_rtype[rtype]
            if scores_from_best_pair:
                summed_scores_for_corpus[rtype]["precision"] += scores_from_best_pair[
                    "precision"
                ]
                summed_scores_for_corpus[rtype]["recall"] += scores_from_best_pair[
                    "recall"
                ]
                summed_scores_for_corpus[rtype]["fmeasure"] += scores_from_best_pair[
                    "fmeasure"
                ]
            # If scores_from_best_pair is None (e.g., all r_chunks had 0 recall, or max_recall remained -1.0),
            # it means this gt_chunk contributes 0.0 to the sums for this rtype,
            # which is implicitly handled as summed_scores_for_corpus started at 0.0.

    # Calculate averages
    if num_gt > 0:
        for rtype in rouge_types:
            final_corpus_scores[rtype]["precision"] = (
                summed_scores_for_corpus[rtype]["precision"] / num_gt
            )
            final_corpus_scores[rtype]["recall"] = (
                summed_scores_for_corpus[rtype]["recall"] / num_gt
            )
            final_corpus_scores[rtype]["fmeasure"] = (
                summed_scores_for_corpus[rtype]["fmeasure"] / num_gt
            )

    return final_corpus_scores
