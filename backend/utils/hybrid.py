import numpy as np
from rank_bm25 import BM25Okapi

class HybridSearcher:
    def __init__(self, metadata_df):
        self.df = metadata_df
        # Use the 'description' column created by the indexer
        self.corpus = [str(d).lower().split() for d in self.df['description'].fillna("").tolist()]
        self.bm25 = BM25Okapi(self.corpus)

    def get_hybrid_scores(self, query_text, query_vec, visual_indices, visual_scores, top_k=10, category_filter=None):
        if not query_text or query_text.strip() == "":
            # If no text, just return visual matches (filtering if needed)
            results = []
            for i, idx in enumerate(visual_indices):
                item = self.df.iloc[idx].to_dict()
                if category_filter and item.get('category', '').lower() != category_filter.lower():
                    continue
                results.append({"metadata": item, "score": visual_scores[i]})
            return results[:top_k]

        query_tokens = query_text.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        if np.max(bm25_scores) > 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)

        final_ranked_results = []
        for i, idx in enumerate(visual_indices):
            v_score = visual_scores[i]
            
            # Safety check: ensure FAISS index matches BM25 corpus size
            if idx >= len(bm25_scores):
                continue

            item = self.df.iloc[idx].to_dict()

            # --- METADATA FILTERING ---
            # If a category is detected (e.g. "ring"), penalize or exclude other categories
            if category_filter:
                item_category = str(item.get('category', '')).lower().strip()
                if item_category != category_filter.lower():
                    continue # Strict filtering: Skip items that don't match
                
            k_score = bm25_scores[idx]
            
            # Adjusted Weights: 40% Visual, 60% Keyword (User Request for "Accuracy++")
            # We favor keywords slightly more to ensure specific terms like "ruby" or "emerald" 
            # aren't overpowered by generic visual similarity.
            total_score = (v_score * 0.4 + (k_score * 0.6))
            
            final_ranked_results.append({
                "metadata": item,
                "score": total_score
            })

        return sorted(final_ranked_results, key=lambda x: x['score'], reverse=True)[:top_k] 