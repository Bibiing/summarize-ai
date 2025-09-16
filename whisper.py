import whisper
import time
import hdbscan
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from collections import defaultdict

try:
    start_time = time.time()
    print("Memuat model Whisper")
    model = whisper.load_model("small")
    print(f"Model dimuat dalam {time.time() - start_time:.2f} detik.")

    start_time = time.time()
    print("Memulai transkripsi file")
    result = model.transcribe("../data/audio/03 - Diagnostic Pre Test Listening Part A Directions.mp3")
    print(f"Transkripsi selesai dalam {time.time() - start_time:.2f} detik.") 

    print(f"Bahasa terdeteksi: {result['language']}")
    print(f"Transkripsi: \n{result['text']}")

except FileNotFoundError:
    print(f"Error: File tidak ditemukan.")
except Exception as e:
    print(f"Terjadi error: {e}")

transcripts = result["text"]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,    # Jumlah karakter per chunk
    chunk_overlap=200   # Karakter yang tumpang tindih untuk menjaga konteks
)

chunks = text_splitter.split_text(transcripts)
print(f"Jumlah potongan: {len(chunks)}")

# if chunks:
#     print(chunks[0])


embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 

# encoding/embedding
print("Membuat vector embeddings untuk setiap chunk")
chunk_embeddings = embedding_model.encode(chunks)
print(f"{len(chunk_embeddings)} embeddings.")

# clustering dengan hdbscan
print("clustering HDBSCAN")
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom')
cluster_labels = clusterer.fit_predict(chunk_embeddings)

# Jumlah cluster yang ditemukan (tidak termasuk noise yang dilabeli -1)
num_discovered_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"HDBSCAN menemukan {num_discovered_clusters} cluster/topik utama.")

# Kelompokkan chunk berdasarkan hasil clustering
clusters = defaultdict(list)
for i, chunk in enumerate(chunks):
    cluster_id = cluster_labels[i]
    # mengabaikan noise
    if cluster_id != -1:
        clusters[cluster_id].append(chunk)

print(f"Teks berhasil dikelompokkan ke dalam {len(clusters)} topik.")

# connect to Gemini API
# try:
#     genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# except Exception as e:
#     print(f"Error: {e}")
#     exit()

# model = genai.GenerativeModel('gemini-1.5-flash')

# # Buat rangkuman untuk setiap cluster topik
# print("\nMembuat rangkuman untuk setiap cluster topik...")
# cluster_summaries = []

# for cluster_id, cluster_chunks in sorted(clusters.items()):
#     full_cluster_text = " " .join(cluster_chunks)
    
#     # Buat prompt untuk merangkum cluster ini
#     prompt = f"""
#     Anda adalah seorang ahli dalam membuat rangkuman. 
#     Berikut adalah potongan teks dari sebuah transkrip yang membahas satu topik spesifik. 
#     Tolong rangkum poin-poin paling penting dari teks ini dalam 2-3 kalimat.

#     Teks:
#     ---
#     {full_cluster_text}
#     ---
#     """
    
#     try:
#         response = model.generate_content(prompt)
#         cluster_summaries.append(response.text)
#         print(f"  -> Rangkuman untuk Cluster {cluster_id} selesai.")
#     except Exception as e:
#         print(f"  -> Gagal merangkum Cluster {cluster_id}. Error: {e}")

# # 2. Tahap "Reduce": Gabungkan semua rangkuman cluster menjadi satu rangkuman akhir
# print("\nMembuat rangkuman final dari semua topik...")

# all_summaries_text = "\n".join(cluster_summaries)

# final_prompt = f"""
# Anda adalah seorang editor ahli.
# Berikut adalah kumpulan rangkuman dari beberapa topik berbeda yang diekstrak dari satu dokumen audio.
# Tolong gabungkan semua poin ini menjadi satu rangkuman akhir yang koheren, lancar, dan mudah dibaca dalam format paragraf.

# Rangkuman per topik:
# ---
# {all_summaries_text}
# ---
# """

# try:
#     final_response = model.generate_content(final_prompt)
    
#     print("\n" + "="*25 + " RANGKUMAN FINAL " + "="*25)
#     print(final_response.text)
#     print("="*68)

# except Exception as e:
#     print(f"Gagal membuat rangkuman final. Error: {e}")