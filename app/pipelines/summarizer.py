import time
from sklearn.cluster import HDBSCAN
from collections import defaultdict
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter # https://python.langchain.com/docs/how_to/recursive_text_splitter/
from sentence_transformers import SentenceTransformer # https://sbert.net/

class Summarizer:
    def __init__(self, gemini_api_key):
        start_time = time.time()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        try:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            print(f"summarizer's ready in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            raise ValueError(f"Gemini API configuration failed: {e}")

    def chunk_text(self, text, max_chunk_size=1500):
        """
        Split the transcribed text into manageable chunks.
        """
        print("Splitting text into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_text(text)
        print(f"number of chunk: {len(chunks)}")
        return chunks

    def cluster_chunks(self, chunks):
        """
        function to create embeddings and cluster chunks by topic using HDBSCAN.
        """
        print("Clustering text chunks")
        start_time = time.time()
        embeddings = self.embedding_model.encode(chunks) # Creating vector embeddings
        # texts with similar meanings will be close to each other

        # Clustering with HDBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
        # different from dbscan, tidak menggunakan eps (radius pencarian)
        # visualize: https://github.com/kcv-if/Modul-ML/blob/main/Modul%201/assets/DBSCAN.gif
        clusterer = HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom') # matriks jarak
        cluster_labels = clusterer.fit_predict(embeddings)
        # cluster_labels = [0, 1, 2, 1, -1, 0, 2]

        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        clustered_chunks = defaultdict(list)
        for i, chunk in enumerate(chunks):
            if cluster_labels[i] != -1:
                clustered_chunks[cluster_labels[i]].append(chunk)

        # If no clusters found (all points labeled as noise), fall back to a single cluster
        if len(clustered_chunks) == 0:
            print("No dense clusters found by HDBSCAN; falling back to a single cluster containing all chunks.")
            clustered_chunks[0] = chunks
            num_clusters = 1

        print(f"Found {num_clusters} main topics.")
        print(f"Clustering completed in {time.time() - start_time:.2f} seconds.")
        return dict(clustered_chunks)
    
    def get_final_summary(self, clusters: dict, language="en"):
        print("Create a summary for each cluster...")
        cluster_summaries = []

        def _extract_text_from_response(response):
            """Robustly extract text from various response shapes returned by the Gemini client."""
            # direct attribute
            try:
                if hasattr(response, "text") and isinstance(response.text, str):
                    return response.text
            except Exception:
                pass

            # dict-like responses
            try:
                if isinstance(response, dict):
                    # common candidate format
                    if "candidates" in response and response["candidates"]:
                        cand = response["candidates"][0]
                        if isinstance(cand, dict):
                            # candidate may contain 'content' or 'text'
                            if "content" in cand:
                                content = cand["content"]
                                if isinstance(content, str):
                                    return content
                                if isinstance(content, dict) and "text" in content:
                                    return content["text"]
                            for k in ("text", "message", "output_text"):
                                if k in cand and isinstance(cand[k], str):
                                    return cand[k]
                    # top-level text keys
                    for key in ("output_text", "text", "message"):
                        if key in response and isinstance(response[key], str):
                            return response[key]
            except Exception:
                pass

            # fallback to stringifying
            try:
                return str(response)
            except Exception:
                return ""

        for cluster_id, cluster_chunks in sorted(clusters.items()):
            full_cluster_text = " ".join(cluster_chunks)
            prompt = (
                f"You are a helpful assistant. Summarize the key points from the following text, "
                f"which is part of an audio transcript. Please provide a concise, one-sentence summary in {language}.\n"
                f"---\nTEXT:\n{full_cluster_text}\n---\nSUMMARY:"
            )            
            try:
                response = self.gemini_model.generate_content(prompt)
                text = _extract_text_from_response(response)
                if text:
                    cluster_summaries.append(text.strip())
                else:
                    print(f"Warning: empty summary text for cluster {cluster_id}; response was: {response}")
                print(f"Summary for Cluster {cluster_id} completed.")
            except Exception as e:
                print(f"Failed to summarize Cluster {cluster_id}. Error: {e}")

        # Reduce Stage
        print("\nCombining all summaries into one...")
        all_summaries_text = "\n".join(cluster_summaries)
        final_prompt = (
            f"You are a professional editor. Combine the following key points from a transcript into a single, coherent paragraph. "
            f"The final summary must be in {language}.\n"
            f"---\nKEY POINTS:\n- {all_summaries_text}\n---\nFINAL SUMMARY PARAGRAPH:"
        )
        try:
            final_response = self.gemini_model.generate_content(final_prompt)
            final_text = _extract_text_from_response(final_response)
            if not final_text:
                print(f"Warning: final summary text empty; response: {final_response}")
                final_text = ""
            return cluster_summaries, final_text
        except Exception as e:
            print(f"Failed to create final summary. Error: {e}")
            return cluster_summaries, "Failed to generate final summary."