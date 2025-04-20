

def compute_recall_at_k(results, relevant_keywords, k):
    top_k = results[:k]
    hits = sum(
        any(kw.lower() in r['assessment_name'].lower() or kw.lower() in r['test_type'].lower()
            for kw in relevant_keywords)
        for r in top_k
    )
    return hits / len(relevant_keywords) if relevant_keywords else 0

def compute_map_at_k(results, relevant_keywords, k):
    top_k = results[:k]
    hits = 0
    sum_precisions = 0

    for i, r in enumerate(top_k):
        is_relevant = any(
            kw.lower() in r['assessment_name'].lower() or kw.lower() in r['test_type'].lower()
            for kw in relevant_keywords
        )
        if is_relevant:
            hits += 1
            sum_precisions += hits / (i + 1)

    return sum_precisions / min(len(relevant_keywords), k) if relevant_keywords else 0
