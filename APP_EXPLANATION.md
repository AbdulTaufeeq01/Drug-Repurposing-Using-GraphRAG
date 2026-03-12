# Drug Repurposing Knowledge Graph Application - Detailed Explanation

## Tools List

### LangChain Agent Tools (Main Tools)
1. `drug_repurposing`
2. `analyze_relationship` / `analyze_relationship_improved`
3. `visualize_graph`
4. `Graph QA` (graph_qa_chain_run)

### Core Processing Functions
5. `load_graph_data`
6. `search_graph_nodes`
7. `search_graph_nodes_enhanced`
8. `get_biomedical_synonyms`
9. `get_candidates`
10. `predict_treatments`
11. `analyze_drug_disease_relationship`
12. `find_direct_relationships`
13. `find_intermediate_nodes`
14. `find_direct_relationships_aql`
15. `find_intermediate_nodes_aql`
16. `extract_subgraph`

### Embedding and Prediction Functions
17. `load_embedding_data`
18. `transE_l2_scoring`

### NLP and Text Processing Functions
19. `get_scispacy_pipeline`
20. `scispacy_synonyms`
21. `extract_disease_name`
22. `generate_explanation`
23. `create_prompt`
24. `graph_to_text`

### Search and Indexing Functions
25. `_build_fuzzy_index`
26. `_fuzzy_candidate_ids`

### Visualization Functions
27. `_create_graph_visualization`
28. `_extract_visualization_request`
29. `_format_node_label`
30. `_format_edge_description`

### Graph Analysis Functions
31. `pagerank_around_node`

### Agent and Chain Functions
32. `graph_qa_chain_run`
33. `integrated_query_handler`
34. `visualize_graph_wrapper`
35. `create_bio_prompt`

### Utility Functions
36. `install_scispacy_model`

## Overview

This application is a comprehensive **Drug Repurposing and Biomedical Knowledge Graph Analysis System** that combines knowledge graph technology, machine learning embeddings, and large language models (LLMs) to identify potential drug repurposing candidates and analyze relationships between biomedical entities.

## Purpose

The application serves multiple purposes:
1. **Drug Repurposing**: Identifies existing drugs that could be repurposed for new diseases
2. **Relationship Analysis**: Analyzes relationships between drugs, diseases, genes, and other biomedical entities
3. **Graph Visualization**: Creates interactive visualizations of knowledge graph structures
4. **Natural Language Querying**: Allows users to query the knowledge graph using natural language

## Architecture

### Core Components

#### 1. **Knowledge Graph Infrastructure**
- **Graph Library**: NetworkX (with optional cuGraph backend for GPU acceleration)
- **Data Sources**: 
  - `nodes.csv`: Contains biomedical entities (drugs, diseases, genes, etc.)
  - `edges.csv`: Contains relationships between entities
- **Graph Statistics**: ~97,238 nodes and ~4.4 million edges

#### 2. **Embedding-Based Prediction System**
- **Model**: TransE L2 (Translation-based embedding model)
- **Embeddings**:
  - Entity embeddings: `DRKG_TransE_l2_entity.npy`
  - Relation embeddings: `DRKG_TransE_l2_relation.npy`
- **Scoring Function**: Uses TransE L2 distance to predict drug-disease treatment relationships
- **Purpose**: Ranks potential drug candidates based on embedding similarity

#### 3. **Natural Language Processing**
- **Biomedical NLP**: scispaCy (`en_core_sci_sm` model) for biomedical text processing
- **Entity Linking**: UMLS (Unified Medical Language System) linker for synonym expansion
- **Fuzzy Matching**: RapidFuzz library for approximate string matching
- **LLM Integration**: OpenAI GPT models (GPT-3.5-turbo, GPT-4o) for:
  - Entity extraction
  - Relationship explanation
  - Query interpretation

#### 4. **LangChain Agent System**
- **Agent Type**: Zero-shot ReAct (Reasoning + Acting) agent
- **Tools Available**:
  1. `drug_repurposing`: Identifies drug candidates for diseases
  2. `analyze_relationship`: Analyzes relationships between two entities
  3. `visualize_graph`: Creates graph visualizations
  4. `Graph QA`: Answers questions about graph relationships

## Key Functions and Workflows

### 1. Graph Loading (`load_graph_data`)
**Purpose**: Loads biomedical knowledge graph from CSV files

**Process**:
- Reads nodes from `nodes.csv` with attributes (name, type, Identifier, etc.)
- Reads edges from `edges.csv` with relationship types
- Creates NetworkX graph structure
- Handles node identifiers in format `Type::ID` (e.g., `Disease::MESH:D045473`)

**Output**: NetworkX Graph object with nodes and edges

### 2. Enhanced Search (`search_graph_nodes_enhanced`)
**Purpose**: Advanced search with biomedical synonym expansion

**Features**:
- **Synonym Expansion**: Uses scispaCy to find biomedical synonyms
- **Fuzzy Matching**: RapidFuzz for approximate string matching
- **Special Handling**: Pre-configured synonyms for common terms (COVID-19, ACE2, diabetes)
- **Scoring System**: Match scores (100 = exact, 90 = case-insensitive, 80 = prefix, 70 = substring)

**Process**:
1. Generate synonyms using `get_biomedical_synonyms()`
2. Build fuzzy index for fast lookup
3. Search across multiple node attributes (name, Identifier, synonyms)
4. Rank results by match score

### 3. Drug Repurposing (`drug_repurposing` tool)
**Purpose**: Identifies potential drug repurposing candidates

**Workflow**:
1. **Disease Extraction**: Uses GPT to extract disease name from query
2. **Disease Matching**: Searches graph for disease nodes using enhanced search
3. **Embedding Lookup**: Maps disease identifiers to embedding indices
4. **Drug Candidate Retrieval**: Loads candidate drugs from `infer_drug.tsv`
5. **Prediction**: Uses `predict_treatments()` to score drug-disease pairs
6. **Relationship Analysis**: Analyzes top candidate's relationship with disease
7. **Explanation Generation**: Uses GPT to explain mechanism of action

**Scoring Method**:
- TransE L2 scoring: `score = gamma - ||drug_emb + treatment_emb - disease_emb||`
- Uses log-sigmoid for probability estimation
- Ranks drugs by predicted treatment score

### 4. Relationship Analysis (`analyze_relationship_improved`)
**Purpose**: Analyzes relationships between two biomedical entities

**Process**:
1. **Entity Extraction**: Uses GPT to extract two entities from query
2. **Entity Matching**: Enhanced search to find entities in graph
3. **Direct Relationships**: Checks for direct edges between entities
4. **Path Finding**: Finds intermediate paths (up to depth 3)
5. **Path Formatting**: Converts paths to readable format
6. **Explanation**: Uses GPT to generate mechanistic explanation

**Optimizations**:
- Direct lookup cache for common entities (ACE2, COVID-19)
- Limited path depth (2-3) for performance
- Early termination when relationships found

### 5. Graph Visualization (`visualize_graph`)
**Purpose**: Creates interactive graph visualizations

**Capabilities**:
- **Full Graph**: Sample visualization of entire graph
- **Subgraph**: Local neighborhood around an entity
- **Path Visualization**: Shows paths between two entities

**Features**:
- Color-coded by node type
- Highlighted queried nodes (red)
- Interactive hover information
- Saves to HTML or PNG format

**Visualization Library**: Plotly

### 6. Path Finding Functions

#### `find_direct_relationships`
- Checks for direct edges between two nodes
- Returns edge data and relationship type

#### `find_intermediate_nodes`
- Finds all simple paths between two nodes
- Configurable depth (default: 2)
- Returns paths with vertices and edges
- Performance optimized with early termination

#### `extract_subgraph`
- Extracts subgraph containing drug, disease, and connecting paths
- Used for focused analysis

### 7. Embedding Functions

#### `load_embedding_data`
- Loads entity and relation embeddings from numpy files
- Creates bidirectional mappings (name ↔ ID)
- Handles TSV mapping files

#### `transE_l2_scoring`
- Computes TransE L2 distance score
- Formula: `gamma - ||drug_emb + treatment_emb - disease_emb||`
- Lower distance = higher score = better match

#### `predict_treatments`
- Scores all drug-disease pairs
- Aggregates scores across multiple treatment relations
- Returns top-k predictions with drug names

## Data Files

### Required Input Files:
1. **`nodes.csv`**: Node data with attributes
2. **`edges.csv`**: Edge/relationship data
3. **`embed/DRKG_TransE_l2_entity.npy`**: Entity embeddings
4. **`embed/DRKG_TransE_l2_relation.npy`**: Relation embeddings
5. **`embed/entities.tsv`**: Entity ID mapping
6. **`embed/relations.tsv`**: Relation ID mapping
7. **`infer_drug.tsv`**: Candidate drug list
8. **`drugbank vocabulary.csv`**: DrugBank ID to common name mapping

## Usage Examples

### Example 1: Drug Repurposing
```python
result = integrated_query_handler("What are possible drug repurposing candidates for Covid 19")
```
- Extracts "COVID-19" as disease
- Finds disease node in graph
- Predicts top drug candidates using embeddings
- Analyzes relationship for top candidate
- Returns explanation with mechanism of action

### Example 2: Relationship Analysis
```python
result = integrated_query_handler("Analyze the relationship between ACE2 and coronavirus")
```
- Extracts "ACE2" and "coronavirus" as entities
- Finds both entities in graph
- Discovers direct and indirect relationships
- Generates mechanistic explanation

### Example 3: Graph Visualization
```python
result = integrated_query_handler("visualize the relationship between diabetes and Glyburide")
```
- Identifies visualization request
- Finds entities in graph
- Extracts paths between entities
- Creates interactive Plotly visualization

## Technical Details

### Performance Optimizations
1. **Caching**: 
   - scispaCy pipeline cached with `@lru_cache`
   - Fuzzy index cached per graph instance
2. **Early Termination**: Path finding stops when sufficient results found
3. **Limited Depth**: Path searches limited to depth 2-3
4. **Batch Processing**: Embedding operations use vectorized numpy operations

### Error Handling
- Graceful fallbacks when embeddings missing
- Multiple search strategies (enhanced → regular → direct lookup)
- Clear error messages for missing entities

### Biomedical Term Handling
- Special synonym dictionaries for common terms
- scispaCy UMLS linking for scientific names
- Case-insensitive matching with case preservation

## Dependencies

### Core Libraries:
- **NetworkX**: Graph operations
- **PyTorch**: Embedding operations
- **NumPy/Pandas**: Data manipulation
- **scispaCy**: Biomedical NLP
- **RapidFuzz**: Fuzzy string matching
- **LangChain**: Agent framework
- **OpenAI**: LLM integration
- **Plotly**: Visualization

### Optional:
- **cuGraph**: GPU acceleration (attempted but not required)

## Configuration

### API Keys:
- OpenAI API key stored in `op_pwd` variable (should be moved to environment variable for security)

### File Paths:
All paths configured at top of notebook:
- `NODES_PATH`, `EDGES_PATH`
- `ENTITY_EMB_PATH`, `REL_EMB_PATH`
- `ENTITY_IDMAP_PATH`, `RELATION_IDMAP_PATH`
- `DRUG_LIST_PATH`, `DRUG_VOCAB_PATH`

## Limitations and Future Improvements

### Current Limitations:
1. **Performance**: Large graph (4.4M edges) can be slow for complex queries
2. **Security**: API key hardcoded (should use environment variables)
3. **Path Depth**: Limited to depth 2-3 for performance
4. **Embedding Coverage**: Not all entities have embeddings

### Potential Improvements:
1. Use GPU acceleration (cuGraph) for large graph operations
2. Implement graph database backend (e.g., Neo4j, ArangoDB)
3. Add more sophisticated ranking algorithms
4. Implement caching for frequent queries
5. Add batch processing for multiple diseases
6. Integrate real-time clinical trial data

## Workflow Summary

1. **Initialization**: Load graph and embeddings (6-7 minutes)
2. **Query Processing**: User submits natural language query
3. **Agent Decision**: LangChain agent selects appropriate tool
4. **Tool Execution**: 
   - Search entities in graph
   - Compute predictions/relationships
   - Generate visualizations
5. **Explanation**: GPT generates biological explanation
6. **Output**: Formatted results with visualizations

## Key Innovations

1. **Hybrid Approach**: Combines knowledge graph structure with embedding-based predictions
2. **Enhanced Search**: Biomedical synonym expansion for better entity matching
3. **Multi-tool Agent**: Flexible agent that can handle various query types
4. **Explainable AI**: Generates human-readable explanations of predictions
5. **Interactive Visualization**: Visual exploration of graph relationships

## Use Cases

1. **Drug Discovery**: Identify new uses for existing drugs
2. **Mechanism Research**: Understand drug-disease relationships
3. **Clinical Decision Support**: Provide evidence for treatment options
4. **Research Tool**: Explore biomedical knowledge graph
5. **Education**: Visualize complex biomedical relationships

---

*This application represents a sophisticated integration of knowledge graphs, machine learning, and natural language processing for biomedical research and drug discovery.*

