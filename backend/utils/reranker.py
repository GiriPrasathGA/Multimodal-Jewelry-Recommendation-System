from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        print(f"Loading Reranker model: {model_name}...")
        self.model = CrossEncoder(model_name)
        print("Reranker model loaded.")

    def rerank(self, query, candidates, top_k=12):
        """
        Reranks a list of candidates based on the query.
        """
        if not candidates:
            return []

        # Prepare pairs for Cross-Encoder: [[query, doc_text], [query, doc_text], ...]
        sentence_pairs = []
        for item in candidates:
            # Use description for reranking as it contains the most detail
            desc = item['metadata'].get('description', '')
            sentence_pairs.append([query, desc])

        # Predict scores (Logits)
        scores = self.model.predict(sentence_pairs)

        # Normalize scores using Sigmoid to get 0-1 range
        import numpy as np
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
            
        normalized_scores = sigmoid(scores)

        # Attach sigmoid scores first
        for idx, score in enumerate(normalized_scores):
            candidates[idx]['hybrid_score'] = candidates[idx]['score']
            candidates[idx]['score'] = float(score)

        # Sort by score (Descending) so we know the best candidate
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # Force-High Calibration: Ensure the top result is ~98% (Green)
        # and others scale relative to it, but maintain a high baseline.
        if candidates:
            best_score = candidates[0]['score']
            # Avoid division by zero
            if best_score > 0:
                # We want the best score to appear as 0.98
                # We can straightforwardly map the range [0, best_score] -> [0, 0.98]
                # But to keep it "high", we might want a curve too.
                # Let's try a simple scaling factor first, but boost lower scores too.
                
                # Strategy: "Plateau" Normalization (Hybrid)
                # 1. Linear Scale: Bring best score to 0.98, scale others linearly.
                # 2. Plateau Boost: Ensure the top N items (e.g. 5) stay very high (>0.95) 
                #    if they are reasonably close to the winner, creating a "Top Tier" group.
                
                target = 0.98
                ratio = target / best_score
                
                for idx, item in enumerate(candidates):
                    # 1. Apply Linear Base Scale
                    new_score = item['score'] * ratio
                    
                    # 2. Apply Plateau Boost for Top 5
                    # If it's in the top 5, we push it closer to the max to create a "group of winners"
                    if idx < 5:
                        # Soft boost: Mix the linear score with a high constant
                        # e.g. 80% linear score, 20% fixed high score
                        # actually, let's just clamp the floor for top items if they are decent
                        if new_score > 0.85: # Only boost if it's already decent
                             new_score = max(new_score, 0.96 - (idx * 0.005)) # 0.96, 0.955, 0.95...
                    
                    # Cap at 0.99
                    item['score'] = min(0.99, new_score)
        
        reranked = candidates

        return reranked[:top_k]
