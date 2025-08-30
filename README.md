# FaceFind

FaceFind is a local-first facial recognition and clustering system designed for large personal media libraries.  
It scans photos and videos (frame-by-frame), extracts faces, clusters them by identity, and allows manual tagging to refine accuracy over time.

---

## ✨ Features (MVP)
- **Frame-by-frame analysis** of photos and videos.
- **Face clustering**: group unknown faces into clusters (e.g. Person A, Person B).
- **Manual tagging**: assign names to clusters (e.g. "Alice", "Bob").
- **Iterative feedback loop**: correct mislabels, and FaceFind updates its embeddings.
- **Adjustable strictness threshold**: toggle between conservative (strict matches) and lenient (broader matches).
- **Unknown detection**: flags faces that don’t match any known identity.
- **NAS / external storage support**: process large corpora (1TB+).

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone git@github.com:HarnoorGill24/FaceFind.git
cd FaceFind

2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
3. Install dependencies
(Coming soon: requirements.txt)
pip install -r requirements.txt
⚡ Usage (Planned)
Run the analyzer
python main.py --input /path/to/media/folder --output ./outputs
Review clusters
Clusters will be stored in outputs/clusters/.
Unknown faces will be grouped for manual review.
Tag clusters
Rename cluster folders or provide a labels.json file mapping IDs → names.
Iterate
Re-run FaceFind with updated labels to improve accuracy.
🧑‍💻 Tech Stack
Python 3.11+
PyTorch (MPS backend) — optimized for Apple Silicon (M1/M2/M3/M4).
OpenCV — image & video frame extraction.
FaceNet / ArcFace embeddings — robust face representation.
Scikit-learn — clustering (DBSCAN, HDBSCAN, or k-means).
(Optional) ONNX Runtime — future GPU acceleration.
📂 Project Structure
FaceFind/
│── data/             # Input data (gitignored)
│── outputs/          # Generated clusters & logs (gitignored)
│── src/              # Core source code
│   ├── face_extract.py
│   ├── cluster.py
│   ├── tagger.py
│   └── main.py
│── tests/            # Unit tests
│── requirements.txt
│── .gitignore
│── README.md
🔮 Roadmap
 v1: Core clustering & tagging pipeline
 v2: Web UI for review & tagging
 v3: Active learning (system learns from corrections automatically)
 v4: Packaging as a macOS app (.dmg installer)
⚠️ Disclaimer
This project is for personal use only.
It is not designed for surveillance, security, or law-enforcement applications.