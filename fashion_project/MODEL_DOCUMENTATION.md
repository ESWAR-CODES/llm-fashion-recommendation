# Model Documentation & Technical Specifications

## 1. Overview of Integrated Models

The application integrates three purpose-built model families, structured for modular execution:

| Model Role | Model Identifier | Input Modality | Vector Dim | Weights Size | Hardware Requirement |
|---|---|---|:---:|:---:|---|
| **Text Retrieval** | `sentence-transformers/all-MiniLM-L6-v2` | Product Text / Query | 384-d | ~80 MB | CPU compatible (instant inference) |
| **Visual Retrieval**| `sentence-transformers/clip-ViT-B-32` | Product Images / Text | 512-d | ~600 MB | CPU compatible (offline indexing) |
| **Language Model** | `Qwen/Qwen2.5-3B-Instruct` | ChatML Conversation | N/A | ~6.2 GB | GPU recommended (Colab T4); CPU Fallback |

---

## 2. Sentence Transformers: all-MiniLM-L6-v2

### 2.1 Model Profile
- **Hugging Face Hub:** `sentence-transformers/all-MiniLM-L6-v2`
- **Underlying Architecture:** 6-layer MiniLM Siamese BERT network
- **Output Embedding Dimension:** 384 float32 numbers
- **Maximum Sequence Length:** 256 tokens
- **Normalization:** Strictly $L_2$ unit-normalized ($\|\mathbf{v}\|_2 = 1.0$)

### 2.2 Text Representation Format
Product descriptions are constructed using a fixed deterministic template:
```
Name: {productDisplayName}
Type: {articleType} | Color: {baseColour} | Usage: {usage}
Category: {masterCategory} | Season: {season}
```

### 2.3 Precomputed Embedding Matrix
- **Artifact:** `artifacts/embeddings.npy`
- **Shape:** `(10000, 384)`
- **File Size:** 15.36 MB
- **Data Type:** `numpy.float32`
- **Memory Footprint:** ~15 MB RAM when loaded

---

## 3. CLIP: clip-ViT-B-32

### 3.1 Model Profile
- **Hugging Face Hub:** `sentence-transformers/clip-ViT-B-32` (compatible with `openai/clip-vit-base-patch32`)
- **Visual Backbone:** Vision Transformer (ViT-B/32) with 32x32 patch size
- **Text Backbone:** 12-layer Transformer encoder
- **Joint Projection Dimension:** 512 float32 numbers
- **Image Preprocessing:** 224x224 RGB input, bicubic interpolation, standard ImageNet normalization

### 3.2 Visual Indexing Protocol
- Product images are encoded offline in batches of 128 using `scripts/generate_clip_embeddings.py`.
- Embeddings are unit-normalized ($L_2 = 1.0$) and saved to `artifacts/clip_embeddings.npy` and `artifacts/clip_image_embeddings.npy` (19.53 MB for 10,000 items).
- Row $j$ strictly corresponds to product ID at index $j$ in `artifacts/clip_catalog_ids.json` and `artifacts/catalog_ids.json`.
- Indexing completed on CPU across all 10,000 items (19.3 img/s, 0 failed images, verified by `validate_clip_artifacts()`).

### 3.3 Hardware & Coexistence Rules
- **Offline Indexing:** Precomputed embeddings are reused during web application startup. Zero re-embedding occurs during normal startup.
- **Simultaneous Loading Rule:** The system will never initialize CLIP and Qwen concurrently on hardware with less than 12 GB available RAM / VRAM.

---

## 4. Large Language Model: Qwen2.5-3B-Instruct

### 4.1 Model Profile
- **Hugging Face Hub:** `Qwen/Qwen2.5-3B-Instruct`
- **Parameter Count:** 3.09 Billion parameters
- **Tokenizer Vocabulary:** 151,643 tokens
- **Chat Template:** Official ChatML (`<|im_start|>system...<|im_end|><|im_start|>user...<|im_end|>`)
- **Generation Parameters:** Greedy decoding (`temperature=0.0`, `do_sample=False`, `max_new_tokens=256`)

### 4.2 Structured Output Schema
The LLM emits JSON conforming to the Pydantic `ProposedUpdate` schema:
```json
{
  "reset": false,
  "unsupported": [],
  "updates": [
    {"field": "articleType", "op": "set", "values": ["Shirts"]},
    {"field": "exclude_colors", "op": "add", "values": ["Red"]}
  ]
}
```

### 4.3 Deterministic Fallback on Constrained Hardware
On CPU-only hosting environments without a GPU:
- The `RuleBasedFallbackProvider` activates automatically.
- Uses regex pattern matching and `SYNONYM_MAP` to extract field updates.
- Detects unsupported attributes (size, price, fabric, stock).
- Emits identical `ProposedUpdate` objects with 0ms latency and 0 MB VRAM overhead.
