import time
from sklearn.cluster import HDBSCAN
from collections import defaultdict
from langchain.text_splitter import RecursiveCharacterTextSplitter # https://python.langchain.com/docs/how_to/recursive_text_splitter/
from sentence_transformers import SentenceTransformer # https://sbert.net/

class Summarizer:
    def __init__(self, gemini_model):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.gemini_model = gemini_model

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
        embeddings = self.embedding_model.encode(chunks) # Creating vector embeddings
        # texts with similar meanings will be close to each other

        # Clustering with HDBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
        # different from dbscan, tidak menggunakan eps (radius pencarian)
        # visualize: https://github.com/kcv-if/Modul-ML/blob/main/Modul%201/assets/DBSCAN.gif
        clusterer = HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom') # matriks jarak
        cluster_labels = clusterer.fit_predict(embeddings)
        # cluster_labels = [0, 1, 2, 1, -1, 0, 2]

        clustered_chunks = defaultdict(list)
        noise_count = 0
        for i, label in enumerate(cluster_labels):
            if label != -1:
                clustered_chunks[label].append(chunks[i])
            else:
                noise_count += 1

        num_clusters = len(clustered_chunks)
        total_chunks = len(chunks)
        noise_percentage = (noise_count / total_chunks) * 100 if total_chunks > 0 else 0
        
        print(f"Found {num_clusters} main topics.") 
        print(f"  - Discarded as noise: {noise_count} chunks ({noise_percentage:.1f}%)")

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
            return cluster_summaries, final_response.text
        except Exception as e:
            print(f"Failed to create final summary. Error: {e}")
            return cluster_summaries, "Failed to generate final summary."