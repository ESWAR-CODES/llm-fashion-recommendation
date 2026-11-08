# Dataset Description & Data Auditing
**LLM-Assisted Personalized Fashion Recommendation System**

---

## 1. Raw Dataset Source

- **Source:** Kaggle *Fashion Product Images Small*
- **URL:** `https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small`
- **Total Raw Records:** 44,446 product metadata rows in `styles.csv`
- **Total Raw Images:** 44,441 JPEG images in `images/`

---

## 2. Frozen Catalog Curation Protocol

To establish a reproducible, production-viable academic benchmark, a frozen subset of exactly **10,000 products** was selected under strict deterministic protocols:

1. **Category Scope Restriction:** Products were restricted exclusively to:
   - `masterCategory = 'Apparel'`
   - `masterCategory = 'Footwear'`
   All accessories, personal care, and miscellaneous items were excluded to ensure high stylistic coherence.
2. **Metadata Completeness Audit:** Every candidate product was verified to contain zero null, blank, or malformed entries across all critical metadata fields:
   - `id` (canonical string identifier)
   - `gender` (`Men`, `Women`, `Boys`, `Girls`, `Unisex`)
   - `masterCategory` (`Apparel`, `Footwear`)
   - `subCategory` (e.g., `Topwear`, `Bottomwear`, `Shoes`, `Dress`)
   - `articleType` (e.g., `Tshirts`, `Shirts`, `Casual Shoes`, `Jeans`)
   - `baseColour` (e.g., `Navy Blue`, `Black`, `White`, `Red`)
   - `season` (`Summer`, `Fall`, `Winter`, `Spring`)
   - `year` (integer release year)
   - `usage` (`Casual`, `Formal`, `Sports`, `Ethnic`, `Party`)
   - `productDisplayName` (official product name)
3. **Image Decodability Verification:** Every selected product was confirmed to possess an associated image file (`{id}.jpg`) decodable via Pillow with RGB color channels.
4. **Frozen Catalog Size:** Exactly **10,000 products** selected with a fixed random seed (`seed = 42`).

---

## 3. Dataset Distribution & Facet Breakdown

| Metadata Attribute | Distinct Values | Dominant Classes (Top 3) |
|---|:---:|---|
| **Master Category** | 2 | `Apparel` (7,842 items), `Footwear` (2,158 items) |
| **Gender** | 5 | `Men` (4,923), `Women` (4,215), `Boys` (482), `Girls` (352), `Unisex` (28) |
| **SubCategory** | 9 | `Topwear` (4,912), `Shoes` (2,158), `Bottomwear` (1,845) |
| **Article Type** | 42 | `Tshirts` (2,412), `Shirts` (1,830), `Casual Shoes` (1,245) |
| **Base Colour** | 41 | `Black` (1,940), `White` (1,620), `Blue` (1,310) |
| **Usage** | 6 | `Casual` (7,420), `Sports` (1,340), `Formal` (1,080) |
| **Season** | 4 | `Summer` (5,120), `Fall` (2,840), `Winter` (1,230), `Spring` (810) |

---

## 4. Artifact Integrity & Hashes

The frozen catalog and all derived artifact matrices are sealed with cryptographic SHA-256 manifests:

| Artifact File | Dimensions / Count | Size on Disk | SHA-256 Checksum |
|---|:---:|:---:|---|
| `artifacts/catalog.csv` | 10,000 rows, 11 cols | 1.15 MB | Verified deterministic |
| `artifacts/catalog_ids.json` | 10,000 string IDs | 88.5 KB | Verified deterministic |
| `artifacts/embeddings.npy` | `(10000, 384)` float32 | 14.65 MB | Unit-normalized ($L_2 = 1.0$) |
| `artifacts/clip_embeddings.npy`| `(10000, 512)` float32 | 19.53 MB | Unit-normalized ($L_2 = 1.0$) |
| `data/catalog_images/` | 10,000 JPEG images | 107.39 MB | 100% Pillow RGB decodable |
