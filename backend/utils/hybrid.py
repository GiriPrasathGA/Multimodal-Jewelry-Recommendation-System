import numpy as np
from rank_bm25 import BM25Okapi

class HybridSearcher:
    def __init__(self, metadata_df, reranker=None):
        self.df = metadata_df
        # Use the 'description' column created by the indexer
        self.corpus = [str(d).lower().split() for d in self.df['description'].fillna("").tolist()]
        self.bm25 = BM25Okapi(self.corpus)
        self.reranker = reranker

    def get_hybrid_scores(self, query_text, query_vec, visual_indices, visual_scores, top_k=10, category_filter=None, filters=None):
        # 1. Broad Retrieval Phase
        # If we have a reranker, fetch more items initially (e.g. 100) to give the reranker a good pool
        search_k = 100 if self.reranker else top_k

        if not query_text or query_text.strip() == "":
            # If no text, just return visual matches (filtering if needed)
            results = []
            for i, idx in enumerate(visual_indices):
                if idx >= len(self.df): continue
                item = self.df.iloc[idx].to_dict()
                if category_filter and str(item.get('category', '')).lower() != category_filter.lower():
                    continue
                results.append({"metadata": item, "score": float(visual_scores[i])})
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

            # --- ADVANCED FILTERS ---
            # filters = {"materials": ["Gold"], "gender": ["Women"], "gemstones": ["Diamond"]}
            if filters:
                skip_item = False
                for key, required_values in filters.items():
                    if not required_values: continue
                    
                    # Get item attribute (e.g. "Gold,Silver")
                    # Note: enrich_metadata.py lowercased valid keywords? No, it used capitalized keys but lowercased found?
                    # Let's check enrich_metadata.py: "Gold" -> "Gold" (capitalized).
                    # But the search might send "Gold". 
                    # Let's be case-insensitive to be safe.
                    item_val = str(item.get(key, '')).lower()
                    
                    # Check if ANY of the required values are present
                    # e.g. required=["Gold"], item="silver,gold" -> Match
                    match_found = False
                    for val in required_values:
                        if str(val).lower() in item_val:
                            match_found = True
                            break
                    
                    if not match_found:
                        skip_item = True
                        break
                
                if skip_item:
                    continue
                
            k_score = bm25_scores[idx]
            
            # Adjusted Weights: 40% Visual, 60% Keyword
            total_score = (v_score * 0.4 + (k_score * 0.6))
            
            final_ranked_results.append({
                "metadata": item,
                "score": total_score,
                "id": idx
            })

        # Sort by initial hybrid score
        final_ranked_results.sort(key=lambda x: x['score'], reverse=True)
        
        # 2. Reranking Phase
        if self.reranker and query_text.strip():
            # Slice top candidates for reranking
            candidates_to_rerank = final_ranked_results[:search_k]
            
            if candidates_to_rerank:
                # Reranker returns sorted list
                reranked_results = self.reranker.rerank(query_text, candidates_to_rerank, top_k=top_k)
                return reranked_results
            
        return final_ranked_results[:top_k] 