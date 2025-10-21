import io
import json
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

st.title("Modèle Word2Vec test streamlit")

# --------- Chargement du modèle (.h5) ---------
@st.cache_resource
def get_model():
    return load_model("word2vec.h5", compile=False)

try:
    model = get_model()
    st.success("Modèle chargé ✅")
except Exception as e:
    st.error(f"Échec du chargement du modèle: {e}")
    st.stop()

# --------- Résumé du modèle ---------
st.subheader("Résumé du modèle")
buf = io.StringIO()
try:
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    st.text(buf.getvalue())
except Exception:
    st.write([l.name for l in model.layers])

# --------- Récupération des embeddings ---------
embedding_layer = next((l for l in model.layers if l.__class__.__name__.lower() == "embedding"), None)
if embedding_layer is None:
    st.warning("Aucune couche Embedding trouvée dans le modèle. Impossible de calculer les similarités.")
    st.stop()

emb_weights = embedding_layer.get_weights()[0]  # [vocab_size, embedding_dim]
vocab_size, embedding_dim = emb_weights.shape
st.write(f"**Embedding:** vocab_size={vocab_size}, embedding_dim={embedding_dim}")

# --------- Utilitaires cosine ---------
def dot_product(v1, v2):
    return np.sum(v1 * v2)

def cosine_similarity(v1, v2):
    denom = np.sqrt(dot_product(v1, v1) * dot_product(v2, v2))
    return 0.0 if denom == 0 else float(dot_product(v1, v2) / denom)

def find_closest(idx, vectors, topk=10):
    q = vectors[idx]
    sims = np.dot(vectors, q)
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(q)
    sims = np.divide(sims, norms, out=np.zeros_like(sims), where=norms != 0)
    sims[idx] = -np.inf  # exclure lui-même
    top_idx = np.argpartition(-sims, range(topk))[:topk]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(int(i), float(sims[i])) for i in top_idx]

# --------- Vocabulaire (facultatif) ---------
st.subheader("Vocabulaire (facultatif)")
st.caption("Charge un tokenizer (pickle) ou un mapping JSON (word2idx) pour requêter par *mot*.")
tok_file = st.file_uploader("Uploader `tokenizer.pkl` ou `vocab.json`", type=["pkl", "json"])

word2idx, idx2word = None, None
if tok_file is not None and tok_file.name.endswith(".pkl"):
    try:
        tokenizer = pickle.load(tok_file)
        if hasattr(tokenizer, "word_index"):
            word2idx = {w: i for w, i in tokenizer.word_index.items() if i < vocab_size}
            idx2word = {i: w for w, i in word2idx.items()}
            st.success(f"Tokenizer chargé ({len(word2idx)} tokens).")
    except Exception as e:
        st.warning(f"Impossible de lire le tokenizer: {e}")

elif tok_file is not None and tok_file.name.endswith(".json"):
    try:
        mapping = json.load(tok_file)
        if "word2idx" in mapping:
            word2idx = {w: int(i) for w, i in mapping["word2idx"].items() if int(i) < vocab_size}
            idx2word = {i: w for w, i in word2idx.items()}
            st.success(f"Vocabulaire JSON chargé ({len(word2idx)} tokens).")
    except Exception as e:
        st.warning(f"Impossible de lire le JSON: {e}")

# --------- Démo de similarité ---------
st.subheader("Similarité (cosine) dans l’espace d’embedding")
mode = st.radio("Saisir par…", ["Index", "Mot"], horizontal=True)

if mode == "Index":
    idx = st.number_input("Index du mot", min_value=0, max_value=max(0, vocab_size - 1), value=0, step=1)
    topk = st.slider("Nombre de voisins", 5, 30, 10)
    if st.button("Trouver voisins"):
        voisins = find_closest(int(idx), emb_weights, topk)
        if idx2word:
            st.table([{"mot": idx2word.get(i, f"<{i}>"), "index": i, "cosine": s} for i, s in voisins])
        else:
            st.table([{"index": i, "cosine": s} for i, s in voisins])

else:  # par Mot
    if word2idx is None:
        st.info("Charge un tokenizer / vocab pour utiliser le mode 'Mot'.")
    else:
        w = st.text_input("Mot")
        topk = st.slider("Nombre de voisins", 5, 30, 10)
        if st.button("Trouver voisins"):
            if w not in word2idx:
                st.error(f"Mot inconnu dans le vocabulaire: {w}")
            else:
                i = word2idx[w]
                voisins = find_closest(int(i), emb_weights, topk)
                st.table([{"mot": idx2word.get(j, f"<{j}>"), "index": j, "cosine": s} for j, s in voisins])