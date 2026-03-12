# Drug Repurposing Using GraphRAG

<div align="center">

**AI-Powered Drug Repurposing with Knowledge Graph Embeddings and Retrieval-Augmented Generation**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project leverages **Graph-based Retrieval-Augmented Generation (GraphRAG)** to identify promising drug repurposing opportunities. By combining biomedical knowledge graphs with advanced AI techniques, we enhance the drug discovery process with improved explainability and data-driven insights.

**Problem Statement:** Traditional drug development is costly ($2.6B+) and time-consuming (10-15 years). Drug repurposing offers a faster, cheaper alternative by identifying new therapeutic uses for existing drugs.

**Solution:** Our approach uses knowledge graph embeddings and LLM-based generation to rank and explain drug-disease relationships.

## ⭐ Features

✅ **GraphRAG Integration** - Combines knowledge graphs with retrieval-augmented generation  
✅ **Biomedical Knowledge** - Leverages DRKG (Drug Repurposing Knowledge Graph) embeddings  
✅ **Explainability** - AI-generated explanations for drug-disease relationships  
✅ **Scalable Architecture** - Efficient graph traversal and ranking algorithms  
✅ **Interactive UI** - Jupyter notebooks for exploration and analysis  

## 📁 Project Structure

```
drug-repurposing/
├── README.md                          # This file
├── app_notebook.ipynb                # Original exploration notebook
├── app_notebooknx.ipynb              # Main analysis notebook (latest)
├── app_notebooknx_copy.py            # Python script version
├── requirements.txt                   # Python dependencies
├── APP_EXPLANATION.md                # Detailed project explanation
│
├── embed/                             # Pre-trained embeddings
│   ├── DRKG_TransE_l2_entity.npy     # Entity embeddings
│   ├── DRKG_TransE_l2_relation.npy   # Relation embeddings
│   ├── entities.tsv                   # Entity mapping
│   └── relations.tsv                  # Relation mapping
│
├── Data/
│   ├── nodes.csv                      # Knowledge graph nodes
│   ├── edges.csv                      # Knowledge graph edges
│   ├── drugbank vocabulary.csv        # DrugBank entity vocabulary
│   └── infer_drug.tsv                # Test cases for inference
│
└── venv/                              # Python virtual environment
```

## 📦 Prerequisites

- **Python 3.8 or higher**
- **OpenAI API Key** (for GPT-4o integration)
- **pip** (Python package manager)
- 2GB+ RAM (for embeddings loading)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AbdulTaufeeq01/Drug-Repurposing-Using-GraphRAG.git
cd Drug-Repurposing-Using-GraphRAG
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY='your-openai-api-key-here'

# macOS/Linux
export OPENAI_API_KEY='your-openai-api-key-here'
```

⚠️ **Security Note:** Never commit API keys. Use environment variables or `.env` files (added to `.gitignore`).

## 🎬 Quick Start

### Using Jupyter Notebook
```bash
# Start Jupyter
jupyter notebook app_notebooknx.ipynb
```

### Running the Python Script
```bash
python app_notebooknx_copy.py
```

## 📊 Dataset

This project uses the **DRKG (Drug Repurposing Knowledge Graph)**:

| Component | Details |
|-----------|---------|
| **Nodes** | 97,238 entities (drugs, genes, proteins, diseases) |
| **Edges** | 5,874,261 relationships |
| **Embeddings** | TransE model (TranslationBased) with L2 distance |
| **Entity Embedding Dim** | 768 |
| **Relation Embedding Dim** | 768 |
| **Vocabulary** | DrugBank, NCBI, TTD, and other biomedical databases |

### Data Sources
- Drug Repurposing Knowledge Graph (DRKG)
- DrugBank vocabulary mapping
- Biomedical relationship datasets

## 💻 Usage

### Basic Workflow

1. **Load Knowledge Graph Embeddings**
   ```python
   import numpy as np
   
   # Load pre-trained embeddings
   entity_embeddings = np.load('embed/DRKG_TransE_l2_entity.npy')
   relation_embeddings = np.load('embed/DRKG_TransE_l2_relation.npy')
   ```

2. **Query for Drug-Disease Relationships**
   ```python
   # Example: Find drugs for a specific disease
   disease = "Alzheimer's Disease"
   top_drugs = find_repurposing_candidates(disease, top_k=10)
   ```

3. **Generate Explanations**
   ```python
   # AI-powered explanation for why a drug works for a disease
   explanation = generate_explanation(drug, disease, subgraph)
   ```

### Example Queries
- Find new treatments for rare diseases
- Identify drug combinations with synergistic effects
- Explore drug-protein-disease pathways
- Validate potential side effects

## 📈 Results

The system ranks drug candidates based on:
- **Path Relevance** - Shortest paths in the knowledge graph
- **Embedding Similarity** - TransE model predictions
- **Clinical Validity** - Biomedical literature matching
- **Novelty Score** - Already known vs. new repurposing opportunities

Example output:
```
Disease: Type 2 Diabetes

Top 5 Drug Candidates:
1. Metformin       | Score: 0.89 | Known treatment
2. GLP-1 Agonist   | Score: 0.87 | FDA approved
3. SGLT2 Inhibitor | Score: 0.85 | Emerging
4. [New Candidate] | Score: 0.76 | Potential
5. [Novel Drug]    | Score: 0.72 | Needs validation
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Areas for Contribution
- [ ] Additional embedding models (DistMult, ComplEx)
- [ ] Web-based UI for queries
- [ ] Expanded biomedical databases
- [ ] Clinical validation pipeline
- [ ] Performance optimization

## ⚙️ Troubleshooting

| Issue | Solution |
|-------|----------|
| **OPENAI_API_KEY not found** | Ensure environment variable is set: `echo $OPENAI_API_KEY` |
| **Embeddings not loading** | Check file path and available disk space (2GB+) |
| **Out of memory** | Reduce batch size or use GPU acceleration |
| **Jupyter issues** | Try: `pip install --upgrade jupyter` |

## 📚 Additional Resources

- [DRKG Paper](https://arxiv.org/abs/2011.05140)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Knowledge Graphs 101](https://en.wikipedia.org/wiki/Knowledge_graph)
- See [APP_EXPLANATION.md](APP_EXPLANATION.md) for deeper technical details

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Abdul Taufeeq M** - Lead Developer

## 📧 Contact & Support

- **Issues & Questions**: [GitHub Issues](https://github.com/AbdulTaufeeq01/Drug-Repurposing-Using-GraphRAG/issues)
- **Email**: itstaufeeqmdu@gmail.com
- **GitHub**: [@AbdulTaufeeq01](https://github.com/AbdulTaufeeq01)

---

## ⭐ Citation

If you use this project in your research, please cite:

```bibtex
@software{taufeeq2026drug_repurposing,
  title={Drug Repurposing Using GraphRAG},
  author={Taufeeq, Abdul},
  year={2026},
  url={https://github.com/AbdulTaufeeq01/Drug-Repurposing-Using-GraphRAG}
}
```

---

**Last Updated:** March 2026  
**Status:** Active Development
