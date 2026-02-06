import numpy as np
from rank_bm25 import BM25Okapi

class HybridSearcher:
    def __init__(self, metadata_df):
        self.df = metadata_df
        # Use the 'description' column created by the indexer
        self.corpus = [str(d).lower().split() for d in self.df['description'].fillna("").tolist()]
        self.bm25 = BM25Okapi(self.corpus)

    def get_hybrid_scores(self, query_text, query_vec, visual_indices, visual_scores, top_k=10):
        if not query_text or query_text.strip() == "":
            return [{"metadata": self.df.iloc[idx].to_dict(), "score": visual_scores[i]} 
                    for i, idx in enumerate(visual_indices)][:top_k]

        query_tokens = query_text.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        if np.max(bm25_scores) > 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)

        final_ranked_results = []
        for i, idx in enumerate(visual_indices):
            v_score = visual_scores[i]
            k_score = bm25_scores[idx]
            
            # 70% Visual weight, 30% Keyword weight
            total_score = (v_score * 0.7) + (k_score * 0.3)
            
            final_ranked_results.append({
                "metadata": self.df.iloc[idx].to_dict(),
                "score": total_score
            })

        return sorted(final_ranked_results, key=lambda x: x['score'], reverse=True)[:top_k] 