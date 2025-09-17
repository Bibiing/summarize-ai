import time
import hdbscan
from collections import defaultdict
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter # https://python.langchain.com/docs/concepts/text_splitters/
from sentence_transformers import SentenceTransformer

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

    def chunk_text(self, text, max_chunk_size=2000):
        """
        Split the transcribed text into manageable chunks.
        """
        print("Splitting text into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=200
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

        # Clustering with HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom')
        cluster_labels = clusterer.fit_predict(embeddings)
        
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        clustered_chunks = defaultdict(list)
        for i, chunk in enumerate(chunks):
            if cluster_labels[i] != -1:
                clustered_chunks[cluster_labels[i]].append(chunk)

        print(f"Found {num_clusters} main topics.")
        print(f"Clustering completed in {time.time() - start_time:.2f} seconds.")
        return dict(clustered_chunks)
    
    def get_final_summary(self, clusters: dict, language="en"):
        print("Create a summary for each cluster...")
        cluster_summaries = []

        for cluster_id, cluster_chunks in sorted(clusters.items()):
            full_cluster_text = " ".join(cluster_chunks)
            prompt = (
                f"You are a helpful assistant. Summarize the key points from the following text, "
                f"which is part of an audio transcript. Please provide a concise, one-sentence summary in {language}.\n"
                f"---\nTEXT:\n{full_cluster_text}\n---\nSUMMARY:"
            )            
            try:
                response = self.gemini_model.generate_content(prompt)
                cluster_summaries.append(response.text)
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
            return final_response.text
        except Exception as e:
            print(f"Failed to create final summary. Error: {e}")
            return "Failed to generate final summary."