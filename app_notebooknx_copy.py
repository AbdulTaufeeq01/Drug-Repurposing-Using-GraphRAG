#!/usr/bin/env python
# coding: utf-8

# ### Import necessary functions

# In[21]:


import os
import sys
import re
import random
import tempfile
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as fn
import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
from functools import lru_cache
import logging
import spacy
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, Tool, AgentType, initialize_agent, AgentOutputParser
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from IPython.display import display, HTML
from openai import OpenAI
from typing import List, Dict, Optional, Tuple
from itertools import islice


# In[22]:


### Install scispaCy model if not already installed
import subprocess
import sys

def install_scispacy_model():
    """Install the scispaCy model if it's not already available."""
    try:
        import spacy
        nlp = spacy.load("en_core_sci_sm")
        print("scispaCy model 'en_core_sci_sm' is already installed.")
        return True
    except OSError:
        print("Installing scispaCy model 'en_core_sci_sm'...")
        try:
            # Try installing via pip
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "https://github.com/allenai/scispacy/releases/download/v0.5.2/en_core_sci_sm-0.5.2.tar.gz"
            ])
            print("scispaCy model installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install scispaCy model automatically: {e}")
            print("Please install manually by running:")
            print("  python -m pip install https://github.com/allenai/scispacy/releases/download/v0.5.2/en_core_sci_sm-0.5.2.tar.gz")
            return False

# Install the model
install_scispacy_model()


# In[23]:


get_ipython().system(' python -m pip install scispacy')


# ### Change networkx backend to cugraph for hardware acceleration

# In[24]:


logger = logging.getLogger(__name__)

_fuzzy_index_cache = {
    "graph_id": None,
    "node_count": None,
    "choices": [],
    "label_map": {}
}

@lru_cache(maxsize=1)
def get_scispacy_pipeline() -> Optional[spacy.language.Language]:
    """Load and cache the scispaCy pipeline with UMLS linker."""
    try:
        nlp = spacy.load("en_core_sci_sm")
        # Check if scispacy_linker factory is available before trying to add it
        if "scispacy_linker" not in nlp.pipe_names:
            try:
                # Check if the factory exists
                if "scispacy_linker" in nlp.factory_names:
                    nlp.add_pipe(
                        "scispacy_linker",
                        config={"resolve_abbreviations": True, "threshold": 0.85}
                    )
                else:
                    # Linker not available, but pipeline is still usable without it
                    logger.debug("scispacy_linker factory not available. Pipeline will work without entity linking.")
            except Exception as pipe_exc:
                # If adding the pipe fails, continue without it
                logger.debug("Could not add scispacy_linker: %s. Pipeline will work without entity linking.", pipe_exc)
        return nlp
    except Exception as exc:
        logger.warning("scispaCy pipeline unavailable: %s", exc)
        return None

def scispacy_synonyms(term: str) -> List[str]:
    """Generate synonyms via scispaCy entity linker."""
    nlp = get_scispacy_pipeline()
    if not nlp:
        return []

    try:
        doc = nlp(term)
    except Exception as exc:
        logger.warning("Failed to process term '%s' with scispaCy: %s", term, exc)
        return []

    variants: List[str] = []
    linker = None
    if "scispacy_linker" in nlp.pipe_names:
        linker = nlp.get_pipe("scispacy_linker")

    for ent in doc.ents:
        variants.append(ent.text)
        if linker and hasattr(ent._, "kb_ents"):
            for cui, score in ent._.kb_ents[:5]:
                kb_entry = linker.kb.cui_to_entity.get(cui)
                if not kb_entry:
                    continue
                variants.append(kb_entry.canonical_name)
                variants.extend(kb_entry.aliases[:5])
    return variants

def _build_fuzzy_index(graph: nx.Graph) -> None:
    global _fuzzy_index_cache
    if (
        _fuzzy_index_cache["graph_id"] == id(graph)
        and _fuzzy_index_cache["node_count"] == graph.number_of_nodes()
    ):
        return

    label_map: Dict[str, set] = {}
    for node_id, attrs in graph.nodes(data=True):
        labels = {str(node_id)}
        name = attrs.get("name")
        if name:
            labels.add(str(name))
        identifier = attrs.get("Identifier")
        if identifier:
            labels.add(str(identifier))
        synonyms_attr = attrs.get("synonyms")
        if isinstance(synonyms_attr, (list, tuple, set)):
            labels.update(str(item) for item in synonyms_attr if item)

        for label in labels:
            normalized = label.strip().lower()
            if not normalized:
                continue
            label_map.setdefault(normalized, set()).add(node_id)

    _fuzzy_index_cache = {
        "graph_id": id(graph),
        "node_count": graph.number_of_nodes(),
        "choices": list(label_map.keys()),
        "label_map": label_map,
    }


def _fuzzy_candidate_ids(
    graph: nx.Graph,
    term: str,
    limit: int = 100,
    min_score: int = 65,
) -> List[str]:
    """Return node IDs whose labels approximately match the term."""
    _build_fuzzy_index(graph)
    choices = _fuzzy_index_cache["choices"]
    label_map = _fuzzy_index_cache["label_map"]

    if not choices:
        return []

    matches = rf_process.extract(
        term.lower(),
        choices,
        scorer=rf_fuzz.WRatio,
        score_cutoff=min_score,
        limit=limit,
    )

    candidate_ids: List[str] = []
    for label, score, _ in matches:
        candidate_ids.extend(label_map.get(label, []))

    return list(dict.fromkeys(candidate_ids))



# In[25]:


get_ipython().system('NETWORKX_BACKEND_PRIORITY=cugraph')


# ### Configuration

# In[26]:


NODES_PATH = "nodes.csv"
EDGES_PATH = "edges.csv"
ENTITY_EMB_PATH = r"embed\DRKG_TransE_l2_entity.npy"
REL_EMB_PATH = r"embed\DRKG_TransE_l2_relation.npy"
ENTITY_IDMAP_PATH = r"embed\entities.tsv"
RELATION_IDMAP_PATH = r"embed\relations.tsv"
DRUG_LIST_PATH = "infer_drug.tsv"
DRUG_VOCAB_PATH = "drugbank vocabulary.csv"


# In[27]:


def get_biomedical_synonyms(term: str) -> List[str]:
    """Get biomedical synonyms and variations for a search term."""
    term_lower = term.lower()
    synonyms = [term, term_lower, term.upper(), term.capitalize()]

    # Extend with scispaCy-derived variants
    synonyms.extend(scispacy_synonyms(term))

    # COVID-19 / Coronavirus variations
    if 'covid' in term_lower or 'coronavirus' in term_lower:
        synonyms.extend([
            'COVID-19', 'Covid-19', 'covid-19', 'covid19', 'COVID19',
            'coronavirus', 'Coronavirus', 'CORONAVIRUS',
            'SARS-CoV-2', 'SARS-CoV2', 'sars-cov-2',
            'Coronavirus Infections', 'coronavirus infections',
            'Severe acute respiratory syndrome', 'Severe acute respiratory syndrome-related coronavirus',
            'Middle East Respiratory Syndrome Coronavirus',
            'Coronavirus 229E, Human', 'Coronavirus NL63, Human'
        ])

    # ACE2 variations
    if 'ace2' in term_lower or 'ace 2' in term_lower:
        synonyms.extend(['ACE2', 'ACE 2', 'ace2', 'Angiotensin Converting Enzyme 2'])

    # Diabetes variations
    if 'diabetes' in term_lower:
        synonyms.extend([
            'Diabetes', 'diabetes', 'DIABETES',
            'Diabetes Mellitus', 'diabetes mellitus', 'Diabetes Mellitus, Type 1',
            'Diabetes Mellitus, Type 2', 'Diabetes, Gestational', 'Diabetes Insipidus'
        ])

    return list(dict.fromkeys(synonyms))


def search_graph_nodes_enhanced(
    graph: nx.Graph,
    search_fields: List[str],
    search_term: str,
    filters: Dict = None,
    limit: int = 10,
):
    """Enhanced search with synonym expansion and fuzzy matching."""
    results: List[Dict] = []
    seen_ids = set()

    search_variations = get_biomedical_synonyms(search_term)

    candidate_node_ids = set()
    for variation in search_variations:
        candidate_node_ids.update(_fuzzy_candidate_ids(graph, variation))

    if candidate_node_ids:
        node_iterator = (
            (node_id, graph.nodes[node_id])
            for node_id in candidate_node_ids
            if node_id in graph
        )
    else:
        node_iterator = graph.nodes(data=True)

    for node_id, node_attrs in node_iterator:
        matches = False
        match_score = 0  # Higher score = better match

        for field in search_fields:
            if field not in node_attrs:
                continue

            field_value = str(node_attrs[field])
            field_value_lower = field_value.lower()

            for variation in search_variations:
                variation_lower = variation.lower()

                if variation == field_value:
                    matches = True
                    match_score = max(match_score, 100)
                    break
                if variation_lower == field_value_lower:
                    matches = True
                    match_score = max(match_score, 90)
                    break
                if field_value_lower.startswith(variation_lower):
                    matches = True
                    match_score = max(match_score, 80)
                elif variation_lower in field_value_lower:
                    matches = True
                    match_score = max(match_score, 70)
                elif re.search(r'\b' + re.escape(variation_lower) + r'\b', field_value_lower):
                    matches = True
                    match_score = max(match_score, 75)

            if matches and match_score == 100:
                break

        if not matches:
            continue

        if filters:
            filter_match = True
            for filter_field, filter_value in filters.items():
                if filter_field not in node_attrs or node_attrs[filter_field] != filter_value:
                    filter_match = False
                    break
            if not filter_match:
                continue

        result = {
            '_id': node_id,
            '_key': str(node_id).split('::')[-1] if '::' in str(node_id) else str(node_id),
            'Identifier': node_id,
            '_match_score': match_score,
            **node_attrs,
        }

        if node_id not in seen_ids:
            results.append(result)
            seen_ids.add(node_id)

        if len(results) >= limit * 3:
            break

    results.sort(key=lambda x: x.get('_match_score', 0), reverse=True)

    for result in results[:limit]:
        result.pop('_match_score', None)

    return results[:limit]


# In[28]:


op_pwd = os.getenv('OPENAI_API_KEY', '')  # Use environment variable instead of hardcoded secret
print("API Key configured" if op_pwd else "WARNING: OPENAI_API_KEY not set")


# In[29]:


# Initialize globals used across tool invocations
# These get updated when drug repurposing queries run

disease_list: List[str] = []
clinical_results = pd.DataFrame()


# ### Define functions

# In[30]:


# NOTE: The patching code for _analyze_relationship_core has been moved to Cell 18
# (right after the function definition in Cell 17)
# This cell is kept as a placeholder for reference
pass


# In[31]:


def load_graph_data(nodes_path: str, edges_path: str) -> nx.Graph:
    """
    Load nodes and edges from CSV files and create a NetworkX graph.
    
    Args:
        nodes_path (str): Path to nodes CSV file
        edges_path (str): Path to edges CSV file
    
    Returns:
        nx.Graph: A graph with nodes and edges
    """
    # Read CSV files
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
    
    nodes_df = pd.read_csv(nodes_path)
    
    # Create an empty graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for idx, attrs in nodes_df.iterrows():
        # The Identifier is in the first column (Unnamed: 0)
        # If not found, try 'Identifier' column, then 'id', then use index
        node_id = attrs.get(attrs.index[0]) if len(attrs.index) > 0 and pd.notna(attrs.iloc[0]) else (
            attrs.get('Identifier') if 'Identifier' in attrs else (
                attrs.get('id') if 'id' in attrs else idx
            )
        )
        
        # Create node attributes dictionary, excluding the identifier column
        node_attrs = {}
        for key, value in attrs.items():
            if key != 'Unnamed: 0' and pd.notna(value):
                node_attrs[key] = value
        
        # Extract type from identifier if it's in format "Type::..."
        if isinstance(node_id, str) and '::' in node_id:
            node_type = node_id.split('::')[0]
            node_attrs['type'] = node_type
        
        # Also set Identifier attribute for searching
        node_attrs['Identifier'] = node_id
        
        G.add_node(node_id, **node_attrs)
    
    # Add edges if edges file exists
    if os.path.exists(edges_path):
        edges_df = pd.read_csv(edges_path)
        
        # Add edges with attributes
        for _, row in edges_df.iterrows():
            source = row.get('source', row.get('_from', ''))
            target = row.get('target', row.get('_to', ''))
            
            if source and target:
                attrs = {}
                if 'attributes' in row:
                    attrs = eval(row['attributes']) if isinstance(row['attributes'], str) else row['attributes']
                elif 'Relation' in row:
                    attrs = {'Relation': row['Relation']}
                
                G.add_edge(source, target, **attrs)
    else:
        print(f"Warning: Edges file not found at {edges_path}. Creating graph with nodes only.")
    
    return G

def search_graph_nodes(graph: nx.Graph, search_fields: List[str], 
                       search_term: str, filters: Dict = None, 
                       limit: int = 10):
    """
    Search for nodes in a NetworkX graph by matching attributes.
    
    Args:
        graph: NetworkX graph object
        search_fields: List of node attribute fields to search within
        search_term: Term to search for
        filters: Dictionary of field-value pairs to filter results
        limit: Maximum number of results to return
    
    Returns:
        list: List of matching nodes with their attributes
    """
    results = []
    search_term_lower = search_term.lower()
    
    for node_id, node_attrs in graph.nodes(data=True):
        # Check if node matches search term in any of the search fields
        matches = False
        for field in search_fields:
            if field in node_attrs:
                field_value = str(node_attrs[field]).lower()
                if search_term_lower in field_value:
                    matches = True
                    break
        
        if not matches:
            continue
        
        # Apply filters if provided
        if filters:
            filter_match = True
            for filter_field, filter_value in filters.items():
                if filter_field not in node_attrs or node_attrs[filter_field] != filter_value:
                    filter_match = False
                    break
            if not filter_match:
                continue
        
        # Format result similar to ArangoDB format
        result = {
            '_id': node_id,
            '_key': str(node_id).split('::')[-1] if '::' in str(node_id) else str(node_id),
            'Identifier': node_id,
            **node_attrs
        }
        results.append(result)
        
        if len(results) >= limit:
            break
    
    return results


# In[32]:


# Helper functions for formatting graph elements
def _format_node_label(vertex: Dict) -> str:
    """Format a node/vertex for display."""
    if isinstance(vertex, dict):
        name = vertex.get('name', vertex.get('_id', ''))
        node_id = vertex.get('_id', vertex.get('Identifier', ''))
        if name and name != node_id:
            return f"{name} ({node_id})"
        return str(node_id)
    return str(vertex)

def _format_edge_description(edge: Dict) -> str:
    """Format an edge/relationship for display."""
    if isinstance(edge, dict):
        relation = edge.get('Relation', edge.get('relation', 'related_to'))
        source = edge.get('_from', edge.get('source', 'Unknown'))
        target = edge.get('_to', edge.get('target', 'Unknown'))
        return f"{source} --[{relation}]--> {target}"
    return str(edge)


# In[33]:


def graph_to_text(subgraph, drug_node, disease_node):
    """
    Convert subgraph data into natural language text descriptions using GPT.

    :param subgraph: NetworkX graph object containing the subgraph
    :param drug_node: ID of the drug node
    :param disease_node: ID of the disease node
    :return: Natural language text description
    """
    # Initialize OpenAI client
    client = OpenAI()
    
    # Extract edges and their attributes
    edges = subgraph.edges(data=True)
    descriptions = []

    for edge in edges:
        source, target, attrs = edge
        description = f"{source} is connected to {target} with attributes {attrs}."
        descriptions.append(description)

    # Combine all descriptions into a single text
    combined_text = " ".join(descriptions)
    
    # Create prompt for GPT
    prompt = f"Convert this graph information into a natural language description:\n{combined_text}"
    
    # Use OpenAI API to generate description
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant that converts graph relationships into natural language descriptions."
            }, 
            {
                "role": "user", 
                "content": prompt
            }
        ],
        temperature=1.0,
        max_tokens=150
    )
    
    return response.choices[0].message.content.strip()
def generate_explanation(prompt: str) -> str:
    """
    Generate explanation using OpenAI's language model.

    :param prompt: Prompt text to generate explanation
    :return: Generated explanation text
    """
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o",  # or another appropriate model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that explains medical relationships."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        temperature=1.0,
    )
    return response.choices[0].message.content.strip()

def create_prompt(subgraph, drug_node, disease_node, drug_name=None, disease_name=None) -> str:
    """
    Create prompt for generating explanations from graph context.

    :param subgraph: NetworkX graph object containing the subgraph
    :param drug_node: ID of the drug node
    :param disease_node: ID of the disease node
    :param drug_name: Optional name of the drug node
    :param disease_name: Optional name of the disease node
    :return: Prompt text
    """
    edges = subgraph.edges(data=True)
    descriptions = []

    for edge in edges:
        source, target, attrs = edge
        description = f"{source} is connected to {target} with attributes {attrs}."
        descriptions.append(description)

    combined_text = " ".join(descriptions)
    
    # Use node IDs if names not provided
    drug_display = drug_name if drug_name else drug_node
    disease_display = disease_name if disease_name else disease_node
    
    prompt = (
        f"Explain why {drug_display} is a good treatment for {disease_display} based on the following relationships:\n"
        f"{combined_text}\n"
        f"Remember the name of disease node {disease_node} is {disease_display} and the name of the drug node {drug_node} is {drug_display}. "
        "Provide a detailed explanation."
    )
    return prompt

def extract_disease_name(query, api_key=op_pwd):
    """
    Extract the specific disease name from a medical query.
    
    Parameters:
    -----------
    query : str
        The original query containing a disease name
    api_key : str, optional
        OpenAI API key. Defaults to op_pwd from the global scope.
    
    Returns:
    --------
    str
        Extracted disease name
    """
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a precise medical information extractor. Extract ONLY the specific disease name from the given query. If multiple disease names are present, choose the most specific one."
                },
                {
                    "role": "user", 
                    "content": f"Extract the exact disease name from this query:\n\n{query}"
                }
            ],
            max_tokens=50,
            n=1,
            stop=None,
            temperature=1.0
        )
        
        disease_name = response.choices[0].message.content.strip()
        return disease_name
    
    except Exception as e:
        print(f"Error in extracting disease name: {e}")
        return None


# In[34]:


# Improved analyze_relationship function with enhanced search
# Core function (not a tool) to avoid deprecation warnings
def _analyze_relationship_core(query: str) -> str:
    """Core logic for analyzing relationships between two entities."""
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Extract the two entity names from the relationship query. Return them as 'entity1: name1, entity2: name2'. For biomedical terms, use the most common scientific name (e.g., 'ACE2' not 'ace 2', 'coronavirus' or 'COVID-19' not 'coronavirus disease 2019').",
                },
                {"role": "user", "content": query},
            ],
            temperature=0.7,
        )
        result = response.choices[0].message.content
        parts = [segment.strip() for segment in result.split(',') if ':' in segment]
        if len(parts) < 2:
            return "Could not parse a pair of entities from the query."
        entity1 = parts[0].split(':', 1)[1].strip()
        entity2 = parts[1].split(':', 1)[1].strip()
    except Exception as exc:
        return f"Failed to extract entities from query: {exc}"
    # Use enhanced search with better biomedical term matching
    entity1_results = search_graph_nodes_enhanced(
        graph=G,
        search_fields=["name", "Identifier"],
        search_term=entity1,
        limit=100,
    )
    entity2_results = search_graph_nodes_enhanced(
        graph=G,
        search_fields=["name", "Identifier"],
        search_term=entity2,
        limit=100,
    )
    # Fallback to regular search if enhanced search finds nothing
    if not entity1_results:
        entity1_results = search_graph_nodes_enhanced(
            graph=G,
            search_fields=["name", "Identifier"],
            search_term=entity1,
            limit=100,
        )
    
    if not entity2_results:
        entity2_results = search_graph_nodes_enhanced(
            graph=G,
            search_fields=["name", "Identifier"],
            search_term=entity2,
            limit=100,
        )
    if not entity1_results or not entity2_results:
        return f"Could not find one or both entities in the knowledge graph. Searched for '{entity1}' and '{entity2}'. Try using more specific entity names."
    # Try to find relationships by testing multiple entity combinations
    best_paths = []
    best_entity1 = None
    best_entity2 = None
    best_direct_rels = []
    
    # Test top matches for both entities to find relationships (limited to 2x2 for performance)
    for entity1_node in entity1_results[:2]:
        for entity2_node in entity2_results[:2]:
            entity1_id = entity1_node['Identifier']
            entity2_id = entity2_node['Identifier']
            
            # Check direct relationships first (fast)
            direct_rels = find_direct_relationships(G, entity1_id, entity2_id)
            if direct_rels:
                # If direct relationship found, use it and skip path finding
                if not best_direct_rels:
                    best_direct_rels = direct_rels
                    best_entity1 = entity1_node
                    best_entity2 = entity2_node
                continue
            
            # Check intermediate paths with increasing depth (limited to depth 3 for performance)
            for depth in range(2, 4):  # Reduced max depth to 3
                paths = find_intermediate_nodes(G, entity1_id, entity2_id, max_depth=depth, max_paths=10)
                if paths:
                    if not best_paths or len(best_paths) < len(paths):
                        best_paths = paths
                        best_entity1 = entity1_node
                        best_entity2 = entity2_node
                    # Early termination: if we found paths, no need to search deeper
                    break
            
            # Early termination: if we found both direct and paths, stop searching
            if best_direct_rels and best_paths:
                break
        if best_direct_rels and best_paths:
            break
    
    # Use best matches found, or default to first results
    if best_entity1 and best_entity2:
        entity1_node = best_entity1
        entity2_node = best_entity2
        direct_relationships = best_direct_rels
        intermediate_paths = best_paths if best_paths else []
    else:
        entity1_node = entity1_results[0]
        entity2_node = entity2_results[0]
        entity1_id = entity1_node['Identifier']
        entity2_id = entity2_node['Identifier']
        direct_relationships = find_direct_relationships(G, entity1_id, entity2_id)
        
        # Try deeper search if no relationships found (limited depth for performance)
        intermediate_paths = []
        for depth in range(2, 4):  # Reduced max depth to 3
            intermediate_paths = find_intermediate_nodes(G, entity1_id, entity2_id, max_depth=depth, max_paths=10)
            if intermediate_paths:
                break
    
    entity1_id = entity1_node['Identifier']
    entity2_id = entity2_node['Identifier']
    entity1_label = entity1_node.get('name', entity1_id)
    entity2_label = entity2_node.get('name', entity2_id)
    if not direct_relationships and not intermediate_paths:
        return f"No relationships found between {entity1_label} ({entity1_id}) and {entity2_label} ({entity2_id}). Searched paths up to depth 3."
    response_lines = [f"Analysis of relationship between {entity1_label} and {entity2_label}:"]
    explanation_clauses = []
    if direct_relationships:
        response_lines.append("\\nDirect connections:")
        for rel in direct_relationships:
            edge = rel.get('edge', {}) if isinstance(rel, dict) else rel
            description = _format_edge_description(edge)
            response_lines.append(f"- {description}")
            explanation_clauses.append(description)
    if intermediate_paths:
        response_lines.append("\\nShortest discovered paths:")
        for idx, path in enumerate(intermediate_paths[:5], start=1):
            node_labels = [_format_node_label(vertex) for vertex in path.get('vertices', [])]
            edge_descriptions = [_format_edge_description(edge) for edge in path.get('edges', [])]
            response_lines.append(f"Path {idx}: {' -> '.join(node_labels)}")
            for edge_desc in edge_descriptions:
                response_lines.append(f"    • {edge_desc}")
                explanation_clauses.append(edge_desc)
    if explanation_clauses:
        prompt = (
            f"Explain the biological linkage between {entity1_label} and {entity2_label} using the following "
            f"graph-derived observations:\\n" + "\\n".join(explanation_clauses)
        )
        try:
            explanation = generate_explanation(prompt)
            response_lines.append("\\nMechanistic explanation:\\n" + explanation)
        except Exception as exc:
            response_lines.append(f"\\nMechanistic explanation unavailable ({exc}).")
    return "\\n".join(response_lines)

# Tool wrapper that calls the core function
@tool
def analyze_relationship_improved(query: str) -> str:
    """Analyze relationships between two entities mentioned in query with enhanced biomedical term matching."""
    return _analyze_relationship_core(query)

# Alias for backward compatibility
analyze_relationship = analyze_relationship_improved


# In[ ]:


# Patch _analyze_relationship_core to add direct lookups for ACE2 and coronavirus
import functools

# Store original function
_original_analyze_relationship_core = _analyze_relationship_core

@functools.wraps(_original_analyze_relationship_core)
def _analyze_relationship_core_patched(query: str) -> str:
    """Patched version with direct lookups for common entities."""
    # Direct identifier lookup for common biomedical entities
    direct_lookups = {
        'ACE2': 'Gene::59272',
        'ace2': 'Gene::59272',
        'ACE 2': 'Gene::59272',
        'Angiotensin Converting Enzyme 2': 'Gene::59272',
        'coronavirus': 'Disease::MESH:D045473',
        'COVID-19': 'Disease::MESH:D045473',
        'COVID19': 'Disease::MESH:D045473',
        'covid-19': 'Disease::MESH:D045473',
        'SARS-CoV-2': 'Disease::MESH:D045473',
        'SARS-CoV2': 'Disease::MESH:D045473',
        'sars-cov-2': 'Disease::MESH:D045473',
    }
    
    # Extract entities using original logic
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Extract the two entity names from the relationship query. Return them as 'entity1: name1, entity2: name2'. For biomedical terms, use the most common scientific name (e.g., 'ACE2' not 'ace 2', 'coronavirus' or 'COVID-19' not 'coronavirus disease 2019').",
                },
                {"role": "user", "content": query},
            ],
            temperature=0.7,
        )
        result = response.choices[0].message.content
        parts = [segment.strip() for segment in result.split(',') if ':' in segment]
        if len(parts) < 2:
            return "Could not parse a pair of entities from the query."
        entity1 = parts[0].split(':', 1)[1].strip()
        entity2 = parts[1].split(':', 1)[1].strip()
    except Exception as exc:
        return f"Failed to extract entities from query: {exc}"
    
    # Check direct lookups first
    if entity1 in direct_lookups:
        direct_id = direct_lookups[entity1]
        if direct_id in G:
            # Create a modified query that uses the direct ID
            entity1_results = [{
                'Identifier': direct_id,
                'name': G.nodes[direct_id].get('name', entity1),
                **G.nodes[direct_id]
            }]
        else:
            entity1_results = search_graph_nodes_enhanced(
                graph=G,
                search_fields=["name", "Identifier"],
                search_term=entity1,
                limit=100,
            )
    else:
        entity1_results = search_graph_nodes_enhanced(
            graph=G,
            search_fields=["name", "Identifier"],
            search_term=entity1,
            limit=100,
        )
    
    if entity2 in direct_lookups:
        direct_id = direct_lookups[entity2]
        if direct_id in G:
            entity2_results = [{
                'Identifier': direct_id,
                'name': G.nodes[direct_id].get('name', entity2),
                **G.nodes[direct_id]
            }]
        else:
            entity2_results = search_graph_nodes_enhanced(
                graph=G,
                search_fields=["name", "Identifier"],
                search_term=entity2,
                limit=100,
            )
    else:
        entity2_results = search_graph_nodes_enhanced(
            graph=G,
            search_fields=["name", "Identifier"],
            search_term=entity2,
            limit=100,
        )
    
    if not entity1_results or not entity2_results:
        return f"Could not find one or both entities in the knowledge graph. Searched for '{entity1}' and '{entity2}'. Try using more specific entity names."
    
    # Continue with relationship finding (use reduced depth for performance)
    best_paths = []
    best_entity1 = None
    best_entity2 = None
    best_direct_rels = []
    
    # Test top matches (reduced to 2x2 and depth 2 for speed)
    for entity1_node in entity1_results[:2]:
        for entity2_node in entity2_results[:2]:
            entity1_id = entity1_node['Identifier']
            entity2_id = entity2_node['Identifier']
            
            # Check direct relationships first
            direct_rels = find_direct_relationships(G, entity1_id, entity2_id)
            if direct_rels:
                if not best_direct_rels:
                    best_direct_rels = direct_rels
                    best_entity1 = entity1_node
                    best_entity2 = entity2_node
                continue
            
            # Check intermediate paths with reduced depth
            for depth in range(2, 3):  # Depth 2 only for speed
                paths = find_intermediate_nodes(G, entity1_id, entity2_id, max_depth=depth, max_paths=5)
                if paths:
                    if not best_paths or len(best_paths) < len(paths):
                        best_paths = paths
                        best_entity1 = entity1_node
                        best_entity2 = entity2_node
                    break
            
            if best_direct_rels and best_paths:
                break
        if best_direct_rels and best_paths:
            break
    
    # Use best matches or defaults
    if best_entity1 and best_entity2:
        entity1_node = best_entity1
        entity2_node = best_entity2
        direct_relationships = best_direct_rels
        intermediate_paths = best_paths if best_paths else []
    else:
        entity1_node = entity1_results[0]
        entity2_node = entity2_results[0]
        entity1_id = entity1_node['Identifier']
        entity2_id = entity2_node['Identifier']
        direct_relationships = find_direct_relationships(G, entity1_id, entity2_id)
        
        intermediate_paths = []
        for depth in range(2, 3):  # Depth 2 only
            intermediate_paths = find_intermediate_nodes(G, entity1_id, entity2_id, max_depth=depth, max_paths=5)
            if intermediate_paths:
                break
    
    entity1_id = entity1_node['Identifier']
    entity2_id = entity2_node['Identifier']
    entity1_label = entity1_node.get('name', entity1_id)
    entity2_label = entity2_node.get('name', entity2_id)
    
    if not direct_relationships and not intermediate_paths:
        return f"No relationships found between {entity1_label} ({entity1_id}) and {entity2_label} ({entity2_id}). Searched paths up to depth 2."
    
    response_lines = [f"Analysis of relationship between {entity1_label} and {entity2_label}:"]
    explanation_clauses = []
    
    if direct_relationships:
        response_lines.append("\\nDirect connections:")
        for rel in direct_relationships:
            edge = rel.get('edge', {}) if isinstance(rel, dict) else rel
            description = _format_edge_description(edge)
            response_lines.append(f"- {description}")
            explanation_clauses.append(description)
    
    if intermediate_paths:
        response_lines.append("\\nShortest discovered paths:")
        for idx, path in enumerate(intermediate_paths[:5], start=1):
            node_labels = [_format_node_label(vertex) for vertex in path.get('vertices', [])]
            edge_descriptions = [_format_edge_description(edge) for edge in path.get('edges', [])]
            response_lines.append(f"Path {idx}: {' -> '.join(node_labels)}")
            for edge_desc in edge_descriptions:
                response_lines.append(f"    • {edge_desc}")
                explanation_clauses.append(edge_desc)
    
    if explanation_clauses:
        prompt = (
            f"Explain the biological linkage between {entity1_label} and {entity2_label} using the following "
            f"graph-derived observations:\\n" + "\\n".join(explanation_clauses)
        )
        try:
            explanation = generate_explanation(prompt)
            response_lines.append("\\nMechanistic explanation:\\n" + explanation)
        except Exception as exc:
            response_lines.append(f"\\nMechanistic explanation unavailable ({exc}).")
    
    return "\\n".join(response_lines)

# Replace the original function
_analyze_relationship_core = _analyze_relationship_core_patched


# In[35]:


def load_embedding_data(entity_path: str, relation_path: str, 
                        entity_idmap_path: str, relation_idmap_path: str):
    """
    Load embedding data and create mappings.
    
    Args:
        entity_path (str): Path to entity embeddings
        relation_path (str): Path to relation embeddings
        entity_idmap_path (str): Path to entity ID mapping file
        relation_idmap_path (str): Path to relation ID mapping file
    
    Returns:
        Tuple of (entity embeddings, relation embeddings, entity maps, relation map)
    """
    # Load embeddings
    entity_emb = np.load(entity_path)
    rel_emb = np.load(relation_path)
    
    # Create mappings
    entity_map, entity_id_map = {}, {}
    relation_map = {}
    
    with open(entity_idmap_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['name','id'])
        for row_val in reader:
            entity_map[row_val['name']] = int(row_val['id'])
            entity_id_map[int(row_val['id'])] = row_val['name']
    
    with open(relation_idmap_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['name','id'])
        for row_val in reader:
            relation_map[row_val['name']] = int(row_val['id'])
    
    return entity_emb, rel_emb, entity_map, entity_id_map, relation_map
def transE_l2_scoring(drug_emb, treatment_emb, disease_emb, gamma=12.0):
    """
    Compute TransE L2 scoring for drug-disease relationships.
    
    Args:
        drug_emb (torch.Tensor): Drug embeddings
        treatment_emb (torch.Tensor): Treatment relation embeddings
        disease_emb (torch.Tensor): Disease embeddings
        gamma (float, optional): Margin parameter. Defaults to 12.0.
    
    Returns:
        torch.Tensor: Scoring of drug-disease relationships
    """
    score = drug_emb + treatment_emb - disease_emb
    return gamma - torch.norm(score, p=2, dim=-1)
def predict_treatments(drug_ids, disease_ids, treatment_rid, 
                       entity_emb, rel_emb, 
                       entity_id_map, clinical_drug_map, 
                       topk=100):
    """
    Predict potential treatments for given diseases.
    
    Args:
        drug_ids (list): List of drug entity IDs
        disease_ids (list): List of disease entity IDs
        treatment_rid (list): List of treatment relation IDs
        entity_emb (np.ndarray): Entity embeddings
        rel_emb (np.ndarray): Relation embeddings
        entity_id_map (dict): Mapping of entity IDs to names
        clinical_drug_map (dict): Mapping of DrugBank IDs to common names
        topk (int, optional): Number of top predictions to return. Defaults to 100.
    
    Returns:
        pd.DataFrame: DataFrame of predicted treatments
    """
    if not drug_ids or not disease_ids or not treatment_rid:
        return pd.DataFrame(columns=['drug_id', 'score', 'original_index', 'drug_name'])

    drug_indices = [int(idx) for idx in drug_ids if 0 <= int(idx) < len(entity_emb)]
    disease_indices = [int(idx) for idx in disease_ids if 0 <= int(idx) < len(entity_emb)]
    relation_indices = [int(idx) for idx in treatment_rid if 0 <= int(idx) < len(rel_emb)]

    if not drug_indices or not disease_indices or not relation_indices:
        return pd.DataFrame(columns=['drug_id', 'score', 'original_index', 'drug_name'])

    drug_emb = torch.from_numpy(entity_emb[drug_indices]).float()

    all_scores: List[float] = []
    all_drug_indices: List[int] = []

    for relation_idx in relation_indices:
        treatment_emb = torch.from_numpy(rel_emb[relation_idx]).float()
        for disease_idx in disease_indices:
            disease_emb = torch.from_numpy(entity_emb[disease_idx]).float()
            score_tensor = fn.logsigmoid(transE_l2_scoring(drug_emb, treatment_emb, disease_emb))
            all_scores.extend(score_tensor.tolist())
            all_drug_indices.extend(drug_indices)

    if not all_scores:
        return pd.DataFrame(columns=['drug_id', 'score', 'original_index', 'drug_name'])

    predictions_df = pd.DataFrame({
        'drug_index': all_drug_indices,
        'score': all_scores
    })

    if predictions_df.empty:
        return pd.DataFrame(columns=['drug_id', 'score', 'original_index', 'drug_name'])

    predictions_df = predictions_df.sort_values('score', ascending=False)
    predictions_df = predictions_df.drop_duplicates('drug_index', keep='first')
    predictions_df = predictions_df.head(topk).reset_index(drop=True)

    predictions_df['drug_id'] = predictions_df['drug_index'].map(lambda idx: entity_id_map.get(int(idx)))
    predictions_df['drug_name'] = predictions_df['drug_id'].map(clinical_drug_map)
    predictions_df['original_index'] = predictions_df.index

    predictions_df = predictions_df[predictions_df['drug_id'].notnull()]

    return predictions_df[['drug_id', 'score', 'original_index', 'drug_name']]


# In[36]:


def get_candidates(query: str, graph: nx.Graph) -> pd.DataFrame:
    """Return candidate repurposing drugs for a disease query."""
    global disease_list
    global clinical_results

    clinical_results = pd.DataFrame()
    disease_list = []

    disease_name = extract_disease_name(query)
    if not disease_name:
        raise ValueError("Unable to extract a disease name from the query.")

    # Use enhanced search for better matching
    disease_results = search_graph_nodes_enhanced(
        graph=graph,
        search_fields=["name", "Identifier"],
        search_term=disease_name,
        filters={"type": "Disease"},
        limit=10,
    )

    # If no results with filter, try without filter (in case type is not set properly)
    if not disease_results:
        disease_results = search_graph_nodes_enhanced(
            graph=graph,
            search_fields=["name", "Identifier"],
            search_term=disease_name,
            limit=10,
        )
        disease_results = [
            r
            for r in disease_results
            if 'Identifier' in r and isinstance(r['Identifier'], str) and r['Identifier'].startswith('Disease::')
        ]

    if not disease_results:
        raise ValueError(
            f"No disease nodes found matching '{disease_name}'. Please try a different disease term."
        )

    candidate_diseases = [result['Identifier'] for result in disease_results]

    entity_emb, rel_emb, entity_map, entity_id_map, relation_map = load_embedding_data(
        ENTITY_EMB_PATH,
        REL_EMB_PATH,
        ENTITY_IDMAP_PATH,
        RELATION_IDMAP_PATH,
    )

    matched_diseases = [identifier for identifier in candidate_diseases if identifier in entity_map]
    missing_diseases = sorted(set(candidate_diseases) - set(matched_diseases))
    if missing_diseases:
        print(f"Skipping diseases without embeddings: {missing_diseases}")

    if not matched_diseases:
        raise ValueError(f"No embeddings available for disease '{disease_name}'.")

    disease_list = matched_diseases

    candidate_drugs: List[str] = []
    missing_drugs: List[str] = []

    with open(DRUG_LIST_PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='	', fieldnames=['drug', 'ids'])
        for row_val in reader:
            drug_identifier = row_val['drug']
            if drug_identifier in entity_map:
                candidate_drugs.append(drug_identifier)
            else:
                missing_drugs.append(drug_identifier)

    if missing_drugs:
        print(f"Skipping {len(missing_drugs)} drugs without embeddings.")

    if not candidate_drugs:
        raise ValueError("No candidate drugs have corresponding embeddings in the knowledge graph.")

    treatment_relations = ['Hetionet::CtD::Compound:Disease', 'GNBR::T::Compound:Disease']
    missing_relations = [rel for rel in treatment_relations if rel not in relation_map]
    if missing_relations:
        print(f"Skipping missing relation embeddings: {missing_relations}")

    treatment_rid = [relation_map[rel] for rel in treatment_relations if rel in relation_map]
    if not treatment_rid:
        raise ValueError("Required treatment relation embeddings are missing.")

    drug_ids = [entity_map[drug] for drug in candidate_drugs]
    disease_ids = [entity_map[disease] for disease in disease_list]

    drug_vocab = pd.read_csv(DRUG_VOCAB_PATH)
    clinical_drug_map = dict(zip(drug_vocab['DrugBank ID'], drug_vocab['Common name']))
    clinical_drug_map = {k: v for k, v in clinical_drug_map.items() if isinstance(v, str)}

    results_df = predict_treatments(
        drug_ids,
        disease_ids,
        treatment_rid,
        entity_emb,
        rel_emb,
        entity_id_map,
        clinical_drug_map,
    )

    clinical_results = results_df[results_df['drug_name'].notnull()].reset_index(drop=True)
    return clinical_results


# In[37]:


# Deprecated: the improved `get_candidates` implementation lives in the previous cell.
pass


# In[38]:


# Deprecated: see the updated relationship utilities later in the notebook.
pass


# In[39]:


@tool
def drug_repurposing(query: str) -> str:
    """
    Identifies potential drug repurposing candidates for a given disease and analyzes their relationship.

    This tool retrieves potential drug candidates for a specified disease using knowledge graph embeddings 
    and clinical trial data. It ranks the top candidates and analyzes the most relevant drug-disease relationship 
    to provide insights into the mechanism of action.

    Args:
        query (str): A natural language query specifying a disease name and asking for potential drug repurposing candidates.

    Returns:
        str: A formatted string containing a list of predicted drugs in clinical trials and their mechanism of action.
    """
    global disease_list
    global clinical_results
    
    # Retrieve drug candidates
    get_candidates(query, G)
    
    if clinical_results.empty:
        return "No clinical trial drugs found for the given disease."

    # Limit to top 5 drugs
    clinical_results = clinical_results[0:5]
    drug_list = clinical_results['drug_id']
    drug_id = drug_list[0]
    disease_id = disease_list[0]
    try:
    # Analyze drug-disease relationship
        result = analyze_drug_disease_relationship(
            graph=G,
            compound_id=drug_id,  
            disease_id=disease_id  
        )

    # Format output
        output = (
            f"\nDrug Information:\n"
            f"Name: {result['drug_info']['name']}\n"
            f"ID: {result['drug_info']['id']}\n\n"
            f"Disease Information:\n"
            f"Name: {result['disease_info']['name']}\n"
            f"ID: {result['disease_info']['id']}\n\n"
            f"Relationship Description:\n"
            f"{result['description']}\n\n"
            f"Mechanism of Action:\n"
            f"{result['explanation']}"
        )

        return output

    except ValueError as e:
        return f"Error: {e}"


# In[40]:


# Removed duplicate analyze_relationship definition
# Use analyze_relationship_improved or the alias analyze_relationship instead



# In[41]:


def analyze_drug_disease_relationship(graph: nx.Graph, compound_id: str, disease_id: str) -> dict:
    """
    Analyzes the relationship between a compound and disease by extracting the subgraph,
    generating text descriptions and explanations of the mechanism of action.

    Args:
        graph: NetworkX graph object
        compound_id (str): Identifier for the compound/drug
        disease_id (str): Identifier for the disease
    
    Returns:
        dict: Dictionary containing relationship information
    """
    # Get drug node information
    drug_node_results = search_graph_nodes(
        graph=graph,
        search_fields=["Identifier"],
        search_term=compound_id,
        limit=1
    )
    if not drug_node_results:
        raise ValueError(f"No drug found with identifier: {compound_id}")
    
    drug_node_result = drug_node_results[0]
    drug_name = drug_node_result.get('name', compound_id)
    drug_node_id = drug_node_result['Identifier']

    # Get disease node information 
    disease_node_results = search_graph_nodes(
        graph=graph,
        search_fields=["Identifier"], 
        search_term=disease_id,
        limit=1
    )
    if not disease_node_results:
        raise ValueError(f"No disease found with identifier: {disease_id}")

    disease_node_result = disease_node_results[0]
    disease_name = disease_node_result.get('name', disease_id)
    disease_node_id = disease_node_result['Identifier']

    # Extract direct relationships
    direct_rels = find_direct_relationships_aql(graph, drug_node_id, disease_node_id)
    
    # Extract intermediate paths
    paths = find_intermediate_nodes_aql(graph, drug_node_id, disease_node_id)

    # Format the relationships
    relationships = []
    for rel in direct_rels:
        if 'edge' in rel:
            rel_type = rel['edge'].get('Relation', 'Unknown')
            relationships.append({
                'type': 'direct',
                'relation': rel_type,
                'details': rel['edge']
            })

    for path in paths:
        if 'edges' in path and path['edges']:
            for edge in path['edges']:
                rel_type = edge.get('Relation', 'Unknown')
                relationships.append({
                    'type': 'intermediate',
                    'relation': rel_type,
                    'details': edge
                })

    # Generate text description and explanation
    description = f"Analysis of relationship between {drug_name} and {disease_name}:\n"
    for rel in relationships:
        description += f"- {rel['type']} relationship: {rel['relation']}\n"

    prompt = f"Explain why {drug_name} could be effective for treating {disease_name} based on these relationships:\n{description}"
    explanation = generate_explanation(prompt)

    return {
        "drug_info": {
            "id": drug_node_id,
            "name": drug_name
        },
        "disease_info": {
            "id": disease_node_id,
            "name": disease_name
        },
        "relationships": relationships,
        "description": description,
        "explanation": explanation
    }


# In[42]:


def find_direct_relationships(graph: nx.Graph, drug_node: str, disease_node: str):
    """
    Find direct relationships between drug and disease nodes using NetworkX.

    :param graph: NetworkX graph object
    :param drug_node: ID of the drug node
    :param disease_node: ID of the disease node
    :return: List of direct relationships
    """
    direct_rels = []
    
    # Check if there's a direct edge between the nodes
    if graph.has_edge(drug_node, disease_node):
        edge_data = graph.get_edge_data(drug_node, disease_node)
        direct_rels.append({
            'edge': edge_data if edge_data else {},
            'vertex': graph.nodes[disease_node] if disease_node in graph else {}
        })
    
    return direct_rels

def find_intermediate_nodes(graph: nx.Graph, drug_node: str, disease_node: str, max_depth: int = 2, max_paths: Optional[int] = None):
    """
    Find paths with intermediate nodes between drug and disease nodes using NetworkX.

    :param graph: NetworkX graph object
    :param drug_node: ID of the drug node
    :param disease_node: ID of the disease node
    :param max_depth: Maximum path length to search (default: 2)
    :param max_paths: Maximum number of paths to return (default: None, returns all paths)
    :return: List of paths with intermediate nodes
    """
    paths = []
    
    # Check if both nodes exist in the graph
    if drug_node not in graph or disease_node not in graph:
        return paths
    
    # Find all simple paths up to max_depth (use generator for efficiency)
    try:
        # Use generator directly and limit paths early to avoid excessive computation
        path_generator = nx.all_simple_paths(graph, drug_node, disease_node, cutoff=max_depth)
        
        # Limit the number of paths early to improve performance
        max_paths_to_process = max_paths if max_paths is not None and max_paths > 0 else 50  # Default limit of 50
        path_count = 0
        
        for path in path_generator:
            # Early termination if we've found enough paths
            if path_count >= max_paths_to_process:
                break
            path_count += 1
            # Convert path to format similar to ArangoDB
            vertices = []
            edges = []
            
            for i, node_id in enumerate(path):
                node_attrs = graph.nodes[node_id]
                vertices.append({
                    '_id': node_id,
                    **node_attrs
                })
                
                if i < len(path) - 1:
                    edge_data = graph.get_edge_data(path[i], path[i+1])
                    edges.append({
                        '_id': f"{path[i]}/{path[i+1]}",
                        '_from': path[i],
                        '_to': path[i+1],
                        **(edge_data if edge_data else {})
                    })
            
            paths.append({
                'vertices': vertices,
                'edges': edges
            })
            
            # Early termination if we've collected enough paths
            if len(paths) >= max_paths_to_process:
                break
    except (nx.NetworkXNoPath, StopIteration):
        pass
    except Exception as e:
        # Log but don't fail on path finding errors
        import logging
        logging.debug(f"Error finding paths between {drug_node} and {disease_node}: {e}")
    
    return paths

def extract_subgraph(graph: nx.Graph, drug_node: str, disease_node: str):
    """
    Extract subgraph containing the drug, disease, and connecting paths with important metadata.

    :param graph: NetworkX graph object
    :param drug_node: ID of the drug node
    :param disease_node: ID of the disease node
    :return: NetworkX graph object containing the subgraph
    """
    # Find all paths between drug and disease
    paths = find_intermediate_nodes(graph, drug_node, disease_node, max_depth=3)
    
    # Collect all nodes in paths
    all_nodes = set([drug_node, disease_node])
    for path in paths:
        for vertex in path['vertices']:
            all_nodes.add(vertex['_id'])
    
    # Create subgraph
    subgraph = graph.subgraph(all_nodes).copy()
    
    return subgraph

def find_intermediate_nodes_aql(graph: nx.Graph, drug_node: str, disease_node: str):
    """
    Find paths with intermediate nodes between drug and disease nodes (NetworkX version).
    This function replaces the AQL version and maintains the same interface.

    :param graph: NetworkX graph object
    :param drug_node: ID of the drug node
    :param disease_node: ID of the disease node
    :return: List of paths with intermediate nodes
    """
    return find_intermediate_nodes(graph, drug_node, disease_node, max_depth=2)

def find_direct_relationships_aql(graph: nx.Graph, drug_node: str, disease_node: str):
    """
    Find direct relationships between drug and disease nodes (NetworkX version).
    This function replaces the AQL version and maintains the same interface.

    :param graph: NetworkX graph object
    :param drug_node: ID of the drug node
    :param disease_node: ID of the disease node
    :return: List of direct relationships
    """
    return find_direct_relationships(graph, drug_node, disease_node)


# In[43]:


# Removed duplicate analyze_relationship definition
# Use analyze_relationship_improved or the alias analyze_relationship instead



# In[44]:


def _create_graph_visualization(subgraph_nodes=None, subgraph_edges=None, max_nodes=100, 
                            highlight_nodes=None, title="Knowledge Graph Visualization",
                            output_format="html"):
    """
    Create a visualization of the graph or a subgraph using Plotly, display it in a notebook
    and save it to a file.

    Args:
        subgraph_nodes (list, optional): List of node IDs to include in visualization. 
                                        If None, uses full graph (limited by max_nodes).
        subgraph_edges (list, optional): List of (source, target) edges to include.
                                        If None, includes all edges between subgraph_nodes.
        max_nodes (int): Maximum number of nodes to include if using the full graph.
        highlight_nodes (list, optional): List of node IDs to highlight with a larger marker and different color.
        title (str): Title for the visualization.
        output_format (str): Format of the output file ("html" or "png").

    Returns:
        tuple: (Plotly figure object, path to the saved file)
    """
    # Create a color map for node types
    node_types = set()
    for node_id, attrs in G.nodes(data=True):
        if 'type' in attrs:
            node_types.add(attrs['type'])

    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    color_map = {node_type: color_palette[i % len(color_palette)] for i, node_type in enumerate(node_types)}
    
    # Special color for highlighted/queried nodes
    highlight_color = '#FF0000'  # Bright red for highlighted nodes

    nodes_to_viz = set(subgraph_nodes) if subgraph_nodes else set(list(G.nodes())[:max_nodes])
    subgraph = G.subgraph(nodes_to_viz)

    edges_to_viz = subgraph_edges if subgraph_edges else [(u, v, data) for u, v, data in subgraph.edges(data=True)]

    minigraph = nx.Graph()
    for node_id in nodes_to_viz:
        if node_id in G:  # Ensure node exists in original graph
            minigraph.add_node(node_id, **G.nodes[node_id])

    for source, target, data in edges_to_viz:
        if source in minigraph and target in minigraph:  # Ensure both endpoints exist
            minigraph.add_edge(source, target, **(data if isinstance(data, dict) else {}))

    pos = nx.spring_layout(minigraph, seed=42)

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node_id in minigraph.nodes():
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        
        node_attrs = minigraph.nodes[node_id]
        node_name = node_attrs.get('name', node_id)
        node_name = str(node_name)[:17] + "..." if len(str(node_name)) > 20 else str(node_name)
        
        node_type = node_attrs.get('type', 'Unknown')
        node_text.append(f"{node_name}<br>({node_type})")
        
        # Use highlight color for queried nodes, otherwise use the type color
        if highlight_nodes and node_id in highlight_nodes:
            node_color.append(highlight_color)
            node_size.append(25)  # Larger size for highlighted nodes
        else:
            node_color.append(color_map.get(node_type, '#CCCCCC'))
            node_size.append(15)  # Regular size for other nodes

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(showscale=False, color=node_color, size=node_size, line=dict(width=1, color='#888'))
    )

    edge_x, edge_y = [], []
    for source, target, _ in minigraph.edges(data=True):
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    # Add legend entries for node types
    for node_type, color in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=node_type,
            showlegend=True
        ))
    
    # Add legend entry for highlighted/queried nodes
    if highlight_nodes:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=highlight_color),
            name='Queried Node',
            showlegend=True
        ))
    
    # Save to file
    temp_dir = tempfile.gettempdir()
    file_extension = "html" if output_format == "html" else "png"
    output_path = os.path.join(temp_dir, f"graph_viz_{random.randint(1000, 9999)}.{file_extension}")

    try:
        if output_format == "html":
            fig.write_html(output_path)
        elif output_format == "png":
            fig.write_image(output_path, format="png", scale=2)
        
        # Return both the figure (for display) and the path (for reference)
        return fig, output_path
    except Exception as e:
        return None, f"Error generating visualization: {str(e)}"

def _extract_visualization_request(query: str):
    """
    Extract visualization request type and entities using OpenAI.

    Args:
        query (str): The visualization query
        
    Returns:
        tuple: (request_type, entities)
            - request_type: 'full', 'subgraph', or 'path'
            - entities: list of entity names or None
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": """
            Extract visualization request information from the query.
            If the query is about a full graph visualization, return: full, None
            If the query is about a subgraph visualization, return: subgraph, [entity_name]
            If the query is about a path visualization, return: path, [entity1_name, entity2_name]
            """
        }, {
            "role": "user",
            "content": query
        }]
    )

    result = response.choices[0].message.content.strip()
    parts = result.split(',', 1)

    request_type = parts[0].strip().lower()
    entities = None

    if len(parts) > 1 and parts[1].strip().lower() != 'none':
        # Parse the entities part
        entities_str = parts[1].strip()
        
        # Handle list notation if present
        if entities_str.startswith('[') and entities_str.endswith(']'):
            entities_str = entities_str[1:-1]
            
        # Split by comma for multiple entities
        if ',' in entities_str:
            entities = [e.strip() for e in entities_str.split(',')]
        else:
            entities = [entities_str.strip()]

    return request_type, entities

@tool
def visualize_graph(query: str):
    """
    Tool function for visualizing the knowledge graph or subgraphs based on queries.
    Both displays the visualization in the notebook and saves it to a file.

    Args:
        query (str): Query specifying what to visualize. Can be:
                    - "full": Visualize a sample of the full graph
                    - "subgraph [entity name]": Visualize a local subgraph around an entity
                    - "path [entity1] [entity2]": Visualize paths between two entities
        output_format (str): Format to save the file in ("html" or "png")

    Returns:
        Plotly figure object that will display in the notebook
    """
    output_format="html"    
    # Extract request type and entities using OpenAI
    request_type, entities = _extract_visualization_request(query)

    # Handle full graph visualization
    if request_type == "full":
        fig, filepath = _create_graph_visualization(
            max_nodes=50, 
            title="Knowledge Graph Sample", 
            output_format=output_format
        )
        
    # Handle subgraph visualization
    elif request_type == "subgraph" and entities and len(entities) == 1:
        entity_name = entities[0]
        
        # Find the entity in the graph
        entity_nodes = search_graph_nodes(
            graph=G,
            search_fields=["name", "Identifier"],
            search_term=entity_name,
            limit=5
        )
        
        if not entity_nodes:
            return HTML(f"<div style='color:red'>Entity '{entity_name}' not found in the graph.</div>")
        
        entity_id = entity_nodes[0]['Identifier']
        
        # Get the ego network (node and its neighbors)
        if entity_id in G:
            neighbors = list(G.neighbors(entity_id))
            subgraph_nodes = [entity_id] + neighbors
            
            fig, filepath = _create_graph_visualization(
                subgraph_nodes=subgraph_nodes,
                highlight_nodes=[entity_id],
                title=f"Subgraph around {entity_name}",
                output_format=output_format
            )
        else:
            return HTML(f"<div style='color:red'>Entity ID '{entity_id}' not found in the NetworkX graph.</div>")

    # Handle path visualization
    elif request_type == "path" and entities and len(entities) == 2:
        entity1_name = entities[0]
        entity2_name = entities[1]
        
        # Find the entities in the graph
        entity1_nodes = search_graph_nodes(
            graph=G,
            search_fields=["name", "Identifier"],
            search_term=entity1_name,
            limit=5
        )

        entity2_nodes = search_graph_nodes(
            graph=G,
            search_fields=["name", "Identifier"],
            search_term=entity2_name,
            limit=5
        )
        
        if not entity1_nodes:
            return HTML(f"<div style='color:red'>Entity '{entity1_name}' not found.</div>")
        if not entity2_nodes:
            return HTML(f"<div style='color:red'>Entity '{entity2_name}' not found.</div>")
        
        entity1_id = entity1_nodes[0]['Identifier']
        entity2_id = entity2_nodes[0]['Identifier']
        
        # Check if both nodes exist in the graph
        if entity1_id not in G or entity2_id not in G:
            return HTML("<div style='color:red'>One or both entities not found in the NetworkX graph.</div>")
        
        # Find all paths between the nodes (limited to length 3 for performance)
        try:
            paths = list(nx.all_simple_paths(G, entity1_id, entity2_id, cutoff=3))
        except nx.NetworkXNoPath:
            return HTML(f"<div style='color:red'>No paths found between '{entity1_name}' and '{entity2_name}'.</div>")
        
        if not paths:
            return HTML(f"<div style='color:red'>No paths found between '{entity1_name}' and '{entity2_name}'.</div>")
        
        # Collect all nodes and edges in the paths
        path_nodes = set()
        path_edges = []
        
        for path in paths:
            for node in path:
                path_nodes.add(node)
            
            for i in range(len(path) - 1):
                path_edges.append((path[i], path[i+1], G.get_edge_data(path[i], path[i+1])))
        
        fig, filepath = _create_graph_visualization(
            subgraph_nodes=list(path_nodes),
            subgraph_edges=path_edges,
            highlight_nodes=[entity1_id, entity2_id],
            title=f"Paths between {entity1_name} and {entity2_name}",
            output_format=output_format
        )

    else:
        return HTML("""
        <div style='color:red'>
        Could not understand visualization request. Try:<br>
        - 'Show me the full graph'<br>
        - 'Show me a subgraph around [entity]'<br>
        - 'Show me paths between [entity1] and [entity2]'
        </div>
        """)
    
    # If we got a figure and a filepath, display the figure and show a note about the file
    if fig is not None and isinstance(filepath, str):
        # Add a note to the figure's title about the saved file
        fig.update_layout(
            title=f"{fig.layout.title.text} (Saved to: {filepath})"
        )
        
        # In a Jupyter notebook, returning the figure will display it
        return fig
    
    # If we only got an error message
    if isinstance(filepath, str) and filepath.startswith("Error"):
        return HTML(f"<div style='color:red'>{filepath}</div>")
    
    # Fallback return
    return fig


# ### Getting the database ready

# The graph might take about 6-7 minutes to load please wait patiently 

# In[45]:


print("Please wait while we load the knowledge graph data...")
G = load_graph_data(NODES_PATH, EDGES_PATH)
print(f"Graph loaded successfully with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")


# In[ ]:


def pagerank_around_node(target_node,graph = G, depth=1):
    """
    Compute PageRank scores for nodes in the neighborhood of a target node.
    Where traversal and top k node ranking occur
    Performs multi-hop traversal to extract a subgraph around the target node and computes PageRank scores within that subgraph.

    Parameters:
    - graph (networkx.Graph or networkx.DiGraph): The input graph.
    - target_node: The node around which to compute PageRank.
    - depth (int): The depth of the neighborhood (1 for direct neighbors, 2 for neighbors of neighbors, etc.).

    Returns:
    - pagerank_scores (dict): A dictionary of PageRank scores for nodes in the subgraph.
    - subgraph (networkx.Graph or networkx.DiGraph): The subgraph used for the computation.
    """
    # Step 1: Extract the subgraph around the target node
    neighborhood = nx.single_source_shortest_path_length(graph, target_node, depth)
    subgraph = graph.subgraph(neighborhood.keys())

    # Step 2: Compute PageRank on the subgraph
    pagerank_scores = nx.pagerank(subgraph)

    return pagerank_scores, subgraph


# ### Create the agent

# In[47]:


# Create a NetworkX-based GraphQA chain
def graph_qa_chain_run(query: str) -> str:
    """
    Query the knowledge graph using NetworkX and return relevant information.
    
    Args:
        query: Natural language query about the graph
        
    Returns:
        str: Answer based on graph information
    """
    # Extract entities from query using LLM
    client = OpenAI()
    extraction_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Extract entity names from the query. Return them as a comma-separated list."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0.7
    )
    
    entities_str = extraction_response.choices[0].message.content.strip()
    entities = [e.strip() for e in entities_str.split(',')]
    
    # Search for entities in the graph
    found_entities = []
    for entity in entities:
        results = search_graph_nodes(
            graph=G,
            search_fields=["name", "Identifier"],
            search_term=entity,
            limit=3
        )
        found_entities.extend(results)
    
    if not found_entities:
        return "Could not find relevant entities in the knowledge graph."
    
    # Extract relationships and paths
    relationship_info = []
    for i, entity1 in enumerate(found_entities[:3]):
        for entity2 in found_entities[i+1:min(i+3, len(found_entities))]:
            paths = find_intermediate_nodes(
                G,
                entity1['Identifier'],
                entity2['Identifier'],
                max_depth=2
            )
            if paths:
                relationship_info.append({
                    'entity1': entity1.get('name', entity1['Identifier']),
                    'entity2': entity2.get('name', entity2['Identifier']),
                    'paths': paths
                })
    
    # Format the information
    context = f"Query: {query}\n\nFound entities and relationships:\n"
    for rel in relationship_info:
        context += f"\n{rel['entity1']} <-> {rel['entity2']}: Found {len(rel['paths'])} path(s)\n"
    
    # Generate answer using LLM
    answer_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on knowledge graph information."
            },
            {
                "role": "user",
                "content": f"{context}\n\nBased on this information, answer the query: {query}"
            }
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return answer_response.choices[0].message.content.strip()

# Initialize the model
model = ChatOpenAI(temperature=1.0, api_key=op_pwd)

# Store the original visualization function
original_visualize_graph = visualize_graph

# Global variable to store the last visualization
last_visualization = None

# Wrapper for the visualize_graph tool to capture its outputs
def visualize_graph_wrapper(query):
    """Wrapper that captures the visualization output"""
    result = original_visualize_graph(query)
    # Store the last result in a global variable
    global last_visualization
    last_visualization = result
    return result

# Define the tools
tools = [
    drug_repurposing,
    analyze_relationship,
    Tool(
        name="visualize_graph",
        func=visualize_graph_wrapper,
        description="Useful for visualizing the knowledge graph or subgraphs based on queries"
    ),
    Tool(
        name="Graph QA",
        func=graph_qa_chain_run,
        description="Useful for querying the knowledge graph about relationships between entities"
    )
]

# Custom output parser to handle visualization objects
class DirectVisualizationOutputParser(AgentOutputParser):
    def parse(self, llm_output):
        # Check if we have a visualization request
        if "Action: visualize_graph" in llm_output:
            action_input_match = re.search(r'Action Input: "(.*?)"', llm_output)
            if action_input_match:
                visualization_query = action_input_match.group(1)
                return AgentAction(tool="visualize_graph", tool_input=visualization_query, log=llm_output)
       
        # Check if we have a final answer
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
           
        # Standard parsing for other actions
        action_match = re.search(r'Action: (.*?)[\n]', llm_output)
        action_input_match = re.search(r'Action Input: (.*)', llm_output)
       
        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            # Remove quotes if present
            if action_input.startswith('"') and action_input.endswith('"'):
                action_input = action_input[1:-1]
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)
       
        # If no action or final answer is found
        raise ValueError(f"Could not parse LLM output: {llm_output}")

# Initialize the agent with the custom parser
agent = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callback_manager=CallbackManager([BaseCallbackHandler()]),
    verbose=True,
    agent_kwargs={"output_parser": DirectVisualizationOutputParser()}
)

# Function to create biological explanation prompt
def create_bio_prompt(agent_output):
    return f"""
    Based on the following information, explain the relationships
    between the entities in biological terms, focusing on mechanisms,
    pathways, and physiological relevance:
   
    {agent_output}
    """

# Create the final chain for biological interpretations
final_chain = (
    agent
    | RunnablePassthrough.assign(bio_prompt=create_bio_prompt)
    | (lambda x: model.invoke(x["bio_prompt"]).content)
)

# Integrated function that handles both visualization and regular queries
def integrated_query_handler(query):
    """
    Handles queries and returns either visualizations or processed results
    from the final_chain depending on which tool was used.
    """
    # Reset the last visualization
    global last_visualization
    last_visualization = None
    
    # Run the agent
    try:
        # For direct visualization requests, use the agent directly
        if "visualize" in query.lower() or "visualization" in query.lower() or "graph" in query.lower():
            agent_output = agent.invoke({"input": query})
            if isinstance(agent_output, dict):
                agent_output = agent_output.get("output", str(agent_output))
            
            # If a visualization was generated, return it
            if last_visualization is not None:
                return last_visualization
            return agent_output
        else:
            # For other requests, use the final chain with biological interpretation
            return final_chain.invoke({"input": query})
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Usage example
# result = integrated_query_handler("Show me the relationship between ACE2 and COVID-19")


# ### Examples

# In[48]:


result = integrated_query_handler("Analyze the relationship between ACE2 and coronavirus and explain the mechanism")


# In[49]:


print(result)


# In[50]:


result = integrated_query_handler("visualize the relationship between diabetes and Glyburide")

# Display the result
display(result)


# In[51]:


result = integrated_query_handler("What is the relationship between diabetes and Glyburide")


# In[52]:


print(result)


# In[53]:


result = integrated_query_handler("What are possible drug repourposing candidates for Covid 19")


# In[54]:


print(result)


# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=129f09a4-04c4-4e2b-a317-686d4c775f2c' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
