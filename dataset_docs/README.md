# Heart Failure Dataset Documentation

This directory contains documentation related to the collection, screening, and definition of the source dataset used to build the **CGX** (Cardiovascular GraphRAG) system, specifically focused on **Heart Failure**.

To ensure transparency, scientific rigor, and reproducibility according to medical research standards, the documentation is divided into the following sections:

## 1. ICD Code List (`icd_code_list/`)
Defines the clinical scope of heart failure diseases covered by the project based on the International Classification of Diseases (ICD-10).
*   **Purpose:** Establishes the clinical boundaries for the knowledge graph.

## 2. PubMed Queries (`pubmed_queries/`)
Stores the exact cumulative search strings used to retrieve knowledge from PubMed Central (PMC).
*   **Purpose:** Ensures the reproducibility of the source dataset.

## 3. Screening Log (`screening_log/`)
Detailed records of the filtering process (PRISMA-compliant) used to select high-quality clinical evidence from thousands of search results.
*   **Purpose:** Demonstrates the depth and non-biased nature of the data selection.

## 4. Selected Document Identifiers (`selected_documents/`)
A list of PMCID/PMID identifiers for the 50 elite papers integrated into the system.
*   **Purpose:** Provides the original reference catalog.
