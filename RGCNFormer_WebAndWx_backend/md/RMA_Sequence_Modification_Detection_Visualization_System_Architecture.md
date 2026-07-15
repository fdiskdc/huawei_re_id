# RGCNFormer RNA Sequence Modification Detection Visualization System Architecture Document

## Title: RGCNFormer RNA Sequence Modification Detection Visualization System

> **Abstract**: RNA modification is a crucial post-transcriptional regulatory mechanism. Accurate detection of RNA modification sites is of great significance for understanding gene expression regulation. This document provides a systematic architectural description of the RNA Sequence Modification Detection Visualization System based on the Relational Graph Convolutional Network Transformer (RGCNFormer) model, covering four dimensions: requirements analysis, overall system design, detailed system design, and clustering application analysis. The document encompasses functional and non-functional requirements analysis, three-layer architecture design, front-end and back-end detailed design schemes, Redis data model design, and the application of clustering methods in RNA modification detection. Written to undergraduate thesis standards, the document employs Mermaid diagram specifications for architecture diagrams, flowcharts, sequence diagrams, and ER diagrams, aiming to provide complete technical reference for system development, deployment, and maintenance.

> **Keywords**: RNA modification detection; RGCNFormer; Relational Graph Convolutional Network; Transformer; Visualization system; Clustering analysis

---

## Table of Contents

- [Part I: Requirements Analysis](#part-i-requirements-analysis)
  - [1.1 Research Background and Significance](#11-research-background-and-significance)
  - [1.2 Current Research Status](#12-current-research-status)
  - [1.3 System Functional Requirements](#13-system-functional-requirements)
  - [1.4 System Non-Functional Requirements](#14-system-non-functional-requirements)
  - [1.5 Use Case Analysis](#15-use-case-analysis)
- [Part II: Overall System Design](#part-ii-overall-system-design)
  - [2.1 System Architecture Design](#21-system-architecture-design)
  - [2.2 Technology Stack Selection](#22-technology-stack-selection)
  - [2.3 System Module Decomposition](#23-system-module-decomposition)
  - [2.4 Data Flow Design](#24-data-flow-design)
  - [2.5 API Design Overview](#25-api-design-overview)
- [Part III: Detailed System Design](#part-iii-detailed-system-design)
  - [3.1 Frontend Detailed Design](#31-frontend-detailed-design)
  - [3.2 Backend Detailed Design](#32-backend-detailed-design)
- [Part IV: Application Analysis of Clustering in RNA Sequence Modification Detection](#part-iv-application-analysis-of-clustering-in-rna-sequence-modification-detection)
  - [4.1 Overview of Clustering Methods](#41-overview-of-clustering-methods)
  - [4.2 Graph Clustering Mechanism in RGCNFormer](#42-graph-clustering-mechanism-in-rgcnformer)
  - [4.3 UMAP Embedding and Clustering Visualization](#43-umap-embedding-and-clustering-visualization)
  - [4.4 Clustering Analysis in Few-Shot Scenarios](#44-clustering-analysis-in-few-shot-scenarios)
  - [4.5 Clustering Transfer in Zero-Shot Scenarios](#45-clustering-transfer-in-zero-shot-scenarios)
  - [4.6 Ablation Studies and Clustering Performance](#46-ablation-studies-and-clustering-performance)
  - [4.7 Discussion and Future Directions](#47-discussion-and-future-directions)
- [Appendices](#appendices)
- [References](#references)

---

## Part I: Requirements Analysis

This chapter systematically elaborates on the current status and challenges in the field of RNA modification detection, starting from research background and significance. It provides an in-depth analysis of the application prospects and technical advantages of deep learning methods in this field. On this basis, combined with the technical characteristics of the RGCNFormer model, the functional and non-functional requirements of the system are detailed across multiple dimensions including core prediction functions, visualization analysis functions, RNA secondary structure prediction, and multi-platform support. Finally, through use case analysis, the system's user roles, usage scenarios, and interaction processes are clarified, laying a solid foundation for subsequent overall system design and detailed design. As the starting point of the software engineering lifecycle, the quality of requirements analysis directly determines the correctness and completeness of subsequent design and development directions, serving as the programmatic document for the entire system construction.

### 1.1 Research Background and Significance

#### 1.1.1 Biological Background of RNA Modifications

RNA modification, also known as epitranscriptomic modification, is one of the important post-transcriptional regulatory mechanisms, playing a vital role in gene expression regulation, disease development and progression, embryonic development, and epigenetics [1]. To date, over 170 different types of chemical modifications have been discovered in various RNA molecules. These modifications alter the chemical properties of RNA molecules, affecting key biological processes such as RNA stability, translation efficiency, splicing regulation, subcellular localization, and protein binding capacity.

Among the numerous RNA modification types, the following 12 common modifications have important functions in gene regulation and are the primary detection targets of this system:

1. **m6A (N6-methyladenosine)**: The most abundant internal modification in eukaryotic mRNA, accounting for approximately 80% of all methylation modifications. m6A regulates mRNA metabolic fate by recruiting specific reader proteins (e.g., YTHDF family proteins), participating in stem cell differentiation, tumorigenesis, immune responses, and other biological processes [2].

2. **m5C (5-methylcytosine)**: Widely present in tRNA and rRNA, playing an important role in translation fidelity. Recent studies have found that m5C is also prevalent in mRNA, participating in the regulation of mRNA stability and translation efficiency.

3. **Ψ (Pseudouridine)**: One of the most common RNA modifications, enhancing RNA structural stability by strengthening base stacking interactions, distributed in tRNA, rRNA, and snRNA.

4. **ac4C (N4-acetylcytidine)**: A highly conserved RNA acetylation modification catalyzed by the acetyltransferase NAT10, distributed in tRNA, rRNA, and mRNA, affecting translation efficiency and RNA stability.

5. **Am (2'-O-methyladenosine)**, **Cm (2'-O-methylcytidine)**, **Gm (2'-O-methylguanosine)**, **Tm (2'-O-methyluridine)**: Members of the 2'-O-methylation modification family, enhancing RNA chemical stability by adding methyl groups at the ribose 2' position.

6. **m1A (N1-methyladenosine)**: A modification present in tRNA and rRNA that disrupts Watson-Crick base pairing faces, affecting RNA secondary structure.

7. **m6Am (N6,2'-O-dimethyladenosine)**: A methylated derivative of m6A located at the 5' end of mRNA, associated with mRNA stability regulation.

8. **m7G (7-methylguanosine)**: A core component of the mRNA 5' cap structure, critical for mRNA translation initiation and stability.

These modifications coordinate through complex molecular mechanisms, forming a sophisticated epitranscriptomic regulatory network. Traditional experimental detection methods, such as iCLIP (individual-nucleotide resolution Cross-Linking and ImmunoPrecipitation), miCLIP (methylation iCLIP), and SCARLET (Selective Chemical And Ribonucleotide-enrichment Ligation followed by Extension and Termination), although capable of detecting RNA modification sites with high precision, have the following significant limitations:

1. **High cost**: Experimental reagents (e.g., specific antibodies, chemical probes) and specialized equipment require substantial investment, with single experiment costs reaching tens of thousands of RMB, limiting the feasibility of large-scale screening.
2. **Limited throughput**: Each experiment can only detect a limited number of modification sites or specific modification types, making it difficult to meet the high-throughput screening needs at the whole-transcriptome level.
3. **Long cycle**: The entire process from sample preparation, immunoprecipitation, library construction and sequencing to data analysis typically requires several weeks.
4. **High technical barriers**: Requires professional molecular biology experimental skills and expensive sequencing equipment support, limiting its adoption in ordinary laboratories.
5. **False positive issues**: Non-specific antibody binding may lead to false positive results, affecting data reliability.

Therefore, developing efficient computational methods to assist or replace parts of experimental work has become an important research direction in the field of RNA modification detection.

#### 1.1.2 Application Prospects of Deep Learning Methods

With the rapid development of high-throughput sequencing technology and significant improvements in computational power, deep learning-based computational methods have become important tools for large-scale prediction of RNA modification sites [3]. Deep learning methods can automatically learn feature representations from large-scale sequence data, avoiding the tedious manual feature engineering required by traditional machine learning methods.

This project proposes the RGCNFormer (Relational Graph Convolutional Network Transformer) model based on deep learning technology for precise prediction of RNA sequence modification sites. The model innovatively integrates three complementary deep learning techniques to form a multi-scale, multi-level feature extraction framework:

- **Multi-scale Convolutional Neural Network (Multi-scale CNN)**: Extracts sequence patterns of different granularities in parallel using four convolutional kernels with kernel sizes of 1, 3, 5, and 7, capturing single-base features, 3-mer motifs, 5-mer motifs, and 7-mer motifs respectively, achieving multi-level feature extraction from single bases to local sequence patterns.
- **Graph Convolutional Network (GCN)**: Models base pairing relationships in RNA secondary structures as graph structures, using the neighborhood aggregation mechanism of graph convolution to capture spatial topological information of sequences, achieving deep fusion of sequence and structural features.
- **Transformer Class-Query Attention Mechanism**: Defines 12 learnable class query vectors that interact with node features through multi-head attention computation, achieving precise classification of 12 modification types.

Particularly in few-shot and zero-shot learning scenarios, RGCNFormer demonstrates excellent generalization ability, providing new technical solutions for prediction of low-resource modification types (e.g., ac4C, Am), with significant research value and application prospects.

#### 1.1.3 Importance of Visualization Systems

Although deep learning models perform excellently in prediction accuracy, their "black box" nature makes it difficult for researchers to understand the model's decision-making process and internal mechanisms. The RNA Sequence Modification Detection Visualization System is a comprehensive analysis platform built on the RGCNFormer model, aiming to provide researchers with intuitive and efficient tools for RNA modification site prediction and visualization analysis.

Visualization plays an irreplaceable important role in the interpretability research of deep learning models. By presenting the model's internal representations (e.g., attention weights, feature embeddings, graph structures) and decision processes in graphical form, the visualization system can help researchers:

1. **Understand model mechanisms**: Intuitively display sequence regions and base sites that the model focuses on through attention weight heatmaps.
2. **Verify prediction results**: Check whether the model correctly utilizes RNA secondary structure information through GCN graph structure visualization.
3. **Explore data distribution**: Observe distribution patterns of different modification types in feature space through UMAP embedding visualization.
4. **Compare model performance**: Evaluate prediction capabilities and applicable scenarios of different model variants through multi-model comparison functions.

### 1.2 Current Research Status

#### 1.2.1 Computational Methods for RNA Modification Detection

In recent years, various computational tools have been developed for RNA modification site prediction, driving rapid development in this field. According to technical approach, these methods can be classified into the following categories:

1. **Traditional machine learning-based methods**: Such as Random Forest, Support Vector Machine (SVM), etc., which train classifiers by manually designing sequence features (e.g., k-mer frequency, physicochemical properties, secondary structure features). Although achieving certain results, these methods rely on domain expert experience for feature engineering and struggle to capture complex sequence patterns and high-order feature interactions.

2. **Deep learning-based methods**: Such as Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN) and their variants (LSTM, GRU), demonstrating significant advantages in automatic feature extraction. Tools like DeepRMiSite and iPromoter-5mC adopt CNN architectures to automatically learn local patterns in sequences. However, these methods mainly focus on sequence information, with limited utilization of RNA structural information.

3. **Graph neural network-based methods**: Such as Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), etc., which model RNA molecules as graph structures to simultaneously utilize sequence and structural information. RGCNFormer further introduces relational graph convolution and Transformer attention mechanisms on this basis, representing the cutting-edge direction in this field.

#### 1.2.2 Applications of Graph Neural Networks in Biological Sequence Analysis

Graph Neural Networks (GNN) have been widely applied in bioinformatics in recent years [4]. In protein structure prediction, AlphaFold2 achieves atomic-level precision structure prediction using equivariant graph neural networks; in drug molecular design, GNN is used for molecular property prediction and molecular generation. In RNA structure analysis, GNN effectively captures spatial topological information of sequences by modeling base pairing relationships as graph structures.

Relational Graph Convolutional Network (RGCN), as an important variant of GNN, can distinguish different types of base interactions (e.g., Watson-Crick pairing, Wobble pairing) by introducing relation-typed edges, demonstrating unique advantages in handling heterogeneous graph data.

#### 1.2.3 Progress of Transformers in Bioinformatics

Since the Transformer architecture was proposed by Vaswani et al. in 2017 [5], it has achieved breakthrough progress in natural language processing and has gradually been introduced into the bioinformatics field. In protein language models, the ESM series models achieve large-scale protein sequence representation learning based on the Transformer architecture; in genomics, DNABERT uses Transformers to capture long-range dependencies in DNA sequences.

In RNA sequence analysis, the self-attention mechanism of Transformers can effectively capture long-range dependencies in sequences, which is of great significance for understanding RNA's global structure and function. RGCNFormer combines the Transformer's class-query attention mechanism with graph convolutional networks to achieve collaborative modeling of sequence and structural information.

#### 1.2.4 Limitations of Existing Visualization Tools

Most existing RNA modification detection tools lack comprehensive visualization capabilities, making it difficult for users to intuitively understand the model's prediction results and decision basis. Specifically, existing tools have shortcomings in the following areas:

1. **Lack of attention visualization**: Users cannot intuitively understand which sequence regions the model focuses on.
2. **Lack of structural visualization**: RNA secondary structures and graph structure information cannot be presented graphically.
3. **Lack of model comparison functions**: Performance differences between different models are difficult to compare intuitively.
4. **Lack of interactive exploration**: Users cannot interact with visualization results, limiting the depth of data exploration.

This system addresses these shortcomings by providing rich visualization functions including attention weight visualization, GCN graph structure visualization, UMAP embedding visualization, and Integrated Gradients attribution visualization, filling the gap in this field.

### 1.3 System Functional Requirements

#### 1.3.1 Core Prediction Functions

The system's core prediction functions form the foundation of the entire platform, including the following four aspects:

**1. RNA Sequence Submission and Parsing**

The system supports input of six base characters: A, C, G, U, T, N. The T base is automatically converted to U during processing, and the N base represents unknown bases (encoded as [0.25, 0.25, 0.25, 0.25] in one-hot encoding). The system accepts two input methods:

- **Direct input**: Users enter RNA sequence strings directly in the text box, with automatic format validation and abnormal character detection.
- **File upload**: Users upload FASTA format files, with automatic parsing of sequence headers and content.

The sequence length must be no less than 51 nucleotides (nt). Sequences shorter than 1001nt are symmetrically padded (N bases added to both left and right ends), and sequences longer than 1001nt are truncated from the middle to 1001nt.

**2. 12-Class Modification Site Classification Prediction**

The system adopts a multi-label classification strategy, predicting probabilities of 12 modification types for each site in the input sequence. The 12 modification types include: Am, Atol, Cm, Gm, Tm, Ψ (Pseudouridine), ac4C, m1A, m5C, m6A, m6Am, m7G. Grouped by base type into 4 groups: Adenosine (A) group includes Am, Atol, m1A, m6A, m6Am; Cytidine (C) group includes Cm, ac4C, m5C; Guanosine (G) group includes Gm, m7G; Uridine (U) group includes Tm, Ψ. Output includes prediction probabilities (softmax-normalized values), confidence levels, and modification site location information for each modification type.

**3. Single Inference and Batch Inference**

- **Single inference**: Supports real-time prediction of a single RNA sequence, suitable for rapid verification scenarios.
- **Batch inference**: Supports parallel processing of up to 5 sequences (WeChat Mini Program limit), suitable for large-scale screening scenarios. Batch tasks are identified by UUID and support progress queries and result aggregation.

**4. Dual Mode: Synchronous and Asynchronous Inference**

- **Synchronous inference (ONNX Runtime)**: Suitable for real-time prediction of short sequences, with models loaded in ONNX format and using ONNX Runtime for cross-platform inference acceleration, with response times in the order of seconds.
- **Asynchronous inference (Celery task queue)**: Suitable for long sequences or batch tasks, processed in the background through the Celery distributed task queue to avoid blocking user requests. Task status is stored in Redis, supporting PENDING→STARTED→SUCCESS/FAILURE state transitions.

#### 1.3.2 Visualization Analysis Functions

The system's visualization analysis functions are the core feature of this system, comprising 12 visualization components covering from macro-level classification result display to micro-level base-level attribution analysis, forming a complete visualization analysis system. As shown in Table 1.

**Table 1 System Visualization Component List**

| ID | Component Name | Function Description | Visualization Type | Data Interface |
|----|---------------|---------------------|-------------------|---------------|
| 1 | ClassificationViz | 12-class modification probability display | Bar chart / Radar chart | `/api/v1/results/:jobId` |
| 2 | LocalizationViz | Modification site distribution on sequence | Sequence annotation map | `/api/v1/results/:jobId` |
| 3 | AttentionViz | Multi-head attention weight heatmap | Heatmap | `/api/v1/results/:jobId` |
| 4 | AttentionComparisonViz | Attention comparison across modification types | Comparative heatmap | `/api/v1/results/:jobId` |
| 5 | AttentionDistributionViz | Attention weight statistical distribution | Distribution plot | `/api/v1/results/:jobId` |
| 6 | GcnViz | RNA secondary structure graph visualization | Force-directed graph | `/api/v1/results/:jobId` |
| 7 | TargetGcnViz | GCN message passing for specific nodes | Flow diagram | `/api/v1/visualize-gcn-aggregation` |
| 8 | IntegratedGradientsViz | Base-level integrated gradients attribution analysis | Attribution map | `/api/v1/integrated-gradients` |
| 9 | UMapViz | High-dimensional feature UMAP dimensionality reduction visualization | Scatter plot | `/api/v1/umap` |
| 10 | ModelViz | Hierarchical model architecture display | Structure diagram | `/api/v1/model-architecture` |
| 11 | RgcnformerHeatmap | Modification site heatmap | Heatmap | `/api/v1/results/:jobId` |
| 12 | DatasetComparisonHeatmap | Dataset comparison heatmap | Comparative heatmap | `/api/v1/model-comparison` |

Additionally, the ComparePage component supports comparative analysis of multi-model performance metrics, including Accuracy, Precision, Recall, F1-Score, and AUC-ROC evaluation metrics, helping researchers comprehensively evaluate the prediction performance of different models.

#### 1.3.3 RNA Secondary Structure Prediction

RNA secondary structure is key information for understanding RNA function. The system integrates the LinearFold tool [6] for RNA secondary structure prediction. LinearFold uses an approximate algorithm with linear time complexity O(n), employing 5'-to-3' dynamic programming and beam search strategy to handle secondary structure prediction tasks for long sequences in a short time.

Prediction results are output in dot-bracket format, where:
- `(` represents the 5' end (left side) of paired bases;
- `)` represents the 3' end (right side) of paired bases;
- `.` represents unpaired bases (free bases).

Based on secondary structure information, the system automatically constructs graph structures: base pairing relationships (Watson-Crick pairing A-U, G-C and Wobble pairing G-U) are converted to graph edges, with sequential adjacency relationships (phosphodiester bond connections between i and i+1) serving as auxiliary edges, providing structural information for subsequent graph convolution computation.

#### 1.3.4 Multi-Platform Support

The system supports both Web and WeChat Mini Program access methods to meet different usage scenario requirements:

**1. Web Platform**

Provides a complete visualization function suite using the React 19 + TypeScript 5.9 technology stack, combined with Vite 7 build tools and Ant Design 6 component library, implementing a professional data analysis interface. The Web platform includes 10 route pages covering the workspace, result display, 12 visualization components, and model comparison functions.

**2. WeChat Mini Program**

Provides lightweight query and result display functions, developed using native WXML/WXSS based on the glass-easel component framework. The Mini Program supports batch sequence submission (up to 5 sequences), prediction progress polling (querying every 2 seconds), and result display, suitable for mobile quick query scenarios. For complex visualization functions, the Mini Program redirects to the Web platform through an embedded web-view component.

### 1.4 System Non-Functional Requirements

#### 1.4.1 Performance Requirements

- **Single inference response time**: In synchronous inference mode, the prediction response time for a single sequence (1001nt) should be less than 2 seconds to ensure smooth user experience.
- **Batch task throughput**: In asynchronous inference mode, 5 sequences processed in parallel should complete within 30 seconds to meet batch screening needs.
- **Cache hit response**: When Redis cache hits, result return time should be in milliseconds (<50ms), significantly improving response speed for repeated queries.
- **Concurrent user support**: The system should support at least 50 concurrent users simultaneously, achieving horizontal scaling through Gunicorn multi-Worker and Celery distributed task queue.

#### 1.4.2 Usability Requirements

- **Interface friendliness**: Interface design should be clean and intuitive, following Ant Design specifications, with core operation flows not exceeding 3 steps (input sequence → submit → view results).
- **Error handling**: Clear error messages and operation guidance should be provided, including friendly prompts for sequence format errors, server busy status, task timeout, and other exception scenarios.
- **Responsive design**: Adapt to different screen sizes, supporting desktop (1920px+), laptop (1366px+), and mobile (375px+) access.
- **Internationalization support**: Support both Chinese and English languages, with dynamic language switching through LanguageContext.

#### 1.4.3 Scalability Requirements

- **Model hot update**: Support dynamic loading of model files, specifying model paths through configuration files without requiring service restart to update model versions.
- **New modification type extension**: The system architecture supports convenient addition of new modification types, requiring only modifications to category mappings in configuration files and model weight files.
- **New dataset support**: Support dynamic loading of multiple datasets including Human, Plant, ac4C, MultiRM, Gen3, with dataset definitions independent of model code.

#### 1.4.4 Deployment Requirements

- **Containerized deployment**: Adopt Docker containerization technology, orchestrating Flask application services, Celery Worker, and Redis through docker-compose for one-click deployment.
- **Environment isolation**: Development and production environments are isolated, with different environment parameters distinguished through environment variable configuration (`.env` files).
- **Log monitoring**: System runtime status is recorded through Flask and Celery logs, supporting troubleshooting and performance monitoring.

### 1.5 Use Case Analysis

#### 1.5.1 Actor Identification

The system's main actors include the following two user roles:

1. **Researchers**: Bioinformatics researchers who primarily use the system's prediction and visualization functions for RNA modification analysis. Researchers are the core users of the system, with main operations including submitting RNA sequences for prediction, viewing classification results, performing visualization analysis (attention weights, GCN graph structure, UMAP embedding, IG attribution, etc.), and comparing performance of different models.

2. **System Administrators**: Responsible for system deployment, maintenance, and user management. Main operations include managing model versions, monitoring system health status, and managing Redis cache and Celery task queues.

#### 1.5.2 Use Case Diagram

As shown in Figure 1, the system use case diagram illustrates the interaction relationships between different actors and system functions. Researchers can execute 12 use cases covering prediction, visualization, and analysis functional domains; system administrators can execute 2 use cases covering model management and system monitoring.

**Figure 1 System Use Case Diagram**

```mermaid
graph TB
    subgraph SystemBoundary["System Boundary"]
        UC1["Submit RNA Sequence for Prediction"]
        UC2["View 12-Class Modification Classification Results"]
        UC3["View Attention Weight Distribution"]
        UC4["Perform IG Attribution Analysis"]
        UC5["View GCN Graph Structure and Message Passing"]
        UC6["View UMAP Embedding Clustering"]
        UC7["Compare Multi-Model Performance"]
        UC8["View RNA Secondary Structure"]
        UC9["Batch Submit Sequences"]
        UC10["View Prediction Progress"]
        UC11["Manage Model Versions"]
        UC12["Monitor System Health Status"]
    end

    Researcher((Researcher))
    Admin((System Administrator))

    Researcher --> UC1
    Researcher --> UC2
    Researcher --> UC3
    Researcher --> UC4
    Researcher --> UC5
    Researcher --> UC6
    Researcher --> UC7
    Researcher --> UC8
    Researcher --> UC9
    Researcher --> UC10

    Admin --> UC11
    Admin --> UC12

    UC1 -.->|include| UC2
    UC9 -.->|include| UC10
    UC1 -.->|extend| UC8
```

#### 1.5.3 Core Use Case Descriptions

**Use Case 1: Submit RNA Sequence for Prediction**

| Item | Description |
|------|-------------|
| Use Case Name | Submit RNA Sequence for Prediction |
| Use Case ID | UC-001 |
| Actor | Researcher |
| Precondition | User has opened the system workspace page |
| Basic Flow | 1. User enters the workspace page (WorkspacePage); 2. Enters RNA sequence string in the text box or uploads FASTA format file; 3. System automatically validates sequence format (only A/C/G/U/T/N characters, length ≥51nt); 4. User can optionally select target modification type and Top-K parameter; 5. User clicks "Submit" button; 6. System computes SHA256 hash as task ID; 7. System checks Redis cache; if hit, directly returns result; otherwise submits Celery async task; 8. System returns task ID and prediction status |
| Alternative Flow | a. Sequence format error: System prompts "Sequence format incorrect, please check input" and highlights illegal characters; b. Server busy: System prompts "Server busy, please try again later"; c. Insufficient sequence length: System prompts "Sequence length insufficient, minimum 51nt required" |
| Postcondition | Prediction task has been created, user can query prediction results |

**Use Case 2: View 12-Class Modification Classification Results**

| Item | Description |
|------|-------------|
| Use Case Name | View 12-Class Modification Classification Results |
| Use Case ID | UC-002 |
| Actor | Researcher |
| Precondition | Prediction task has completed (status: completed) |
| Basic Flow | 1. User enters the results page (ResultsPage); 2. System retrieves prediction results from Redis cache; 3. System displays 12-class modification prediction probabilities in bar chart; 4. User can hover to view detailed values; 5. User can click to view detailed analysis of specific modification types; 6. Supports result export in CSV/JSON format |
| Alternative Flow | a. Task not completed: System displays progress bar and current status; b. Task failed: System displays error information and retry button |
| Postcondition | User has viewed classification results |

**Use Case 3: Perform IG Attribution Analysis**

| Item | Description |
|------|-------------|
| Use Case Name | Perform Integrated Gradients Attribution Analysis |
| Use Case ID | UC-003 |
| Actor | Researcher |
| Precondition | User has submitted RNA sequence |
| Basic Flow | 1. User enters the IG attribution visualization page; 2. User selects target modification type (targetClassId); 3. System calls Captum library to compute integrated gradients; 4. System displays contribution values of each base site in attribution map; 5. Highlights base sites with highest contribution; 6. User can interactively explore attribution values in different regions |
| Alternative Flow | a. Computation timeout: System prompts "Attribution computation takes longer, please wait patiently" |
| Postcondition | User has viewed attribution analysis results |

---

## Part II: Overall System Design

This chapter elaborates on the overall architecture scheme of the RGCNFormer RNA Sequence Modification Detection Visualization System, starting from system architecture design. It first introduces the three-layer architecture design (User Layer, Service Layer, Model Layer), analyzing the responsibility division and communication mechanisms of each layer. Then it details the technology selection rationale and version information of each technical component. It subsequently describes the system module decomposition strategy and inter-module dependencies. Finally, from the data flow perspective, through top-level, first-level, and second-level data flow diagrams, the complete data flow process within the system is presented, along with an overview of the RESTful API design specifications. The overall system design serves as a bridge connecting requirements analysis and detailed design, with its core goal being to ensure system maintainability, scalability, and performance while meeting functional requirements.

### 2.1 System Architecture Design

#### 2.1.1 Overall Architecture Design

The RGCNFormer RNA Sequence Modification Detection Visualization System adopts a classic three-layer architecture design, as shown in Figure 2. The system is divided into three layers: User Layer, Service Layer, and Model Layer, communicating through standardized RESTful API interfaces to achieve high cohesion and low coupling architectural goals.

**Figure 2 System Overall Architecture Diagram**

```mermaid
graph TB
    subgraph UserLayer["User Layer"]
        Web["Web Frontend<br/>React 19 + TypeScript<br/>Vite 7 + Ant Design 6"]
        WxApp["WeChat Mini Program<br/>Native WXML/WXSS<br/>glass-easel Framework"]
    end

    subgraph ServiceLayer["Service Layer"]
        Flask["Flask API Server<br/>server.py + wsgi.py<br/>Gunicorn WSGI<br/>Flask-CORS Support"]
        Celery["Celery Async Task Queue<br/>tasks.py<br/>Worker Process Pool"]
        Redis["Redis<br/>Message Broker + Result Cache<br/>Session Storage"]
    end

    subgraph ModelLayer["Model Layer"]
        RGCNFormer["RGCNFormer Model<br/>ParallelCNNBlock Multi-scale CNN<br/>GCNBlock Graph Convolution<br/>ClassQueryHead Class-Query Attention"]
        ONNX["ONNX Runtime<br/>Production Inference Engine"]
        LinearFold["LinearFold<br/>RNA Secondary Structure Prediction"]
        Captum["Captum IG<br/>Integrated Gradients Attribution"]
    end

    Web -->|HTTP/REST JSON| Flask
    WxApp -->|HTTP/REST JSON| Flask
    Flask -->|Task Dispatch| Celery
    Flask <-->|Cache Read/Write| Redis
    Celery <-->|Message Broker| Redis
    Celery -->|Model Inference| RGCNFormer
    Flask -->|Synchronous Inference| ONNX
    Celery -->|Secondary Structure Prediction| LinearFold
    Flask -->|Attribution Computation| Captum
```

**Architecture Layer Description**:

- **User Layer**: Includes Web frontend and WeChat Mini Program clients. The Web frontend uses the React 19 + TypeScript 5.9 technology stack, combined with Vite 7 build tools and Ant Design 6 component library, providing a complete analysis function suite with 12 visualization components. The WeChat Mini Program uses native WXML/WXSS development based on the glass-easel component framework, providing lightweight sequence input, result querying, and progress monitoring functions. Both clients communicate with the service layer through unified RESTful API interfaces in JSON format.

- **Service Layer**: Uses Flask as the web application framework, providing production-grade HTTP services through the Gunicorn WSGI server. The service layer integrates the Celery asynchronous task queue and Redis message broker, supporting both synchronous inference (ONNX Runtime direct calls) and asynchronous inference (Celery background tasks) modes. Redis serves triple responsibilities of result caching, task status storage, and WeChat session management, with automatic data expiration through TTL mechanisms.

- **Model Layer**: Contains the RGCNFormer deep learning model (PyTorch implementation), ONNX Runtime inference engine, LinearFold secondary structure prediction tool, and Captum integrated gradients attribution tool. The RGCNFormer model is the core of the system, consisting of three core sub-modules: ParallelCNNBlock (multi-scale CNN module extracting k-mer sequence patterns), GCNBlock (graph convolution module fusing RNA secondary structure information), and ClassQueryHead (class-query attention module achieving 12-class modification classification).

#### 2.1.2 Frontend-Backend Communication Mechanism

The system adopts RESTful API design specifications, with frontend-backend data interaction through HTTP/JSON format. The communication mechanism design follows these principles:

1. **Stateless communication**: Each HTTP request contains complete request information, with the server not storing client session state, facilitating horizontal scaling.
2. **Polling mechanism**: For asynchronous inference tasks, the frontend uses a polling mechanism to periodically query task status from the backend (Web platform uses React Query automatic polling; Mini Program polls manually every 2 seconds).
3. **Cross-origin support**: Through Flask-CORS middleware configuration for cross-origin resource sharing, allowing cross-origin requests from the Web frontend.
4. **Version management**: All API paths contain version numbers (`/api/v1/`), facilitating subsequent interface upgrades and backward compatibility.

### 2.2 Technology Stack Selection

As shown in Table 2, the system's technology selection at each layer follows these principles: **mature and stable** (selecting technologies verified in large-scale production), **active community** (ensuring long-term maintenance and technical support), **excellent performance** (meeting system performance requirements), and **easy maintenance** (reducing later maintenance costs).

**Table 2 System Technology Selection Table**

| Layer | Technology | Version | Selection Rationale |
|-------|-----------|---------|---------------------|
| Web Frontend Framework | React + TypeScript | 19.x / 5.9 | Component-based development, type safety, rich ecosystem, active community |
| Build Tool | Vite | 7.x | ESM-based fast hot update, excellent development experience |
| UI Component Library | Ant Design | 6.x | Enterprise-grade UI component library, unified design specifications, rich components |
| Chart Library | ECharts | 6.x | Baidu open-source statistical chart library, rich chart types, strong interactivity |
| Graph Visualization | AntV G6 + ReactFlow | 5.x / 11.x | Ant Group graph visualization engine + flow chart components, professional-grade graph structure display |
| 3D Rendering | Three.js | 0.182 | WebGL 3D rendering engine, supports 3D molecular structure visualization |
| Relationship Graph | D3.js + react-force-graph | 7.x | Data-Driven Documents + force-directed graph components, flexible custom visualization |
| Data Requests | TanStack React Query | 5.x | Server-side state management, automatic caching, polling, error retry |
| Internationalization | Custom LanguageContext | — | Lightweight i18n solution, supports Chinese-English dynamic switching |
| WeChat Frontend | Native Mini Program | glass-easel | WeChat official component framework, lightweight native, optimal performance |
| Backend Framework | Flask + Flask-CORS | 2.x | Python lightweight web framework, flexible and extensible |
| WSGI Server | Gunicorn | 20.x | Production-grade Python WSGI HTTP server |
| Async Tasks | Celery | 5.x | Python distributed task queue, mature and stable |
| Message Broker | Redis | 7+ | High-performance in-memory database, cache + message broker + session storage |
| Deep Learning | PyTorch + PyG | 1.10+ / 2.0+ | Dynamic computation graph, PyTorch Geometric graph neural network support |
| Inference Engine | ONNX Runtime | — | Microsoft open-source cross-platform inference acceleration engine |
| Structure Prediction | LinearFold | C++ | Linear time complexity RNA secondary structure prediction algorithm |
| Model Interpretability | Captum (IG) | 0.4+ | Facebook open-source model interpretability library, integrated gradients attribution analysis |
| Containerization | Docker + docker-compose | — | Containerized deployment, service orchestration, environment consistency |

### 2.3 System Module Decomposition

As shown in Figure 3, the system module decomposition consists of three major parts: Frontend Display Module, API Gateway Module, and Backend Service Module, with modules communicating through standardized interfaces.

**Figure 3 System Module Relationship Diagram**

```mermaid
graph TB
    subgraph FrontendModule["Frontend Display Module"]
        A["Sequence Input Module<br/>WorkspacePage"]
        B["Result Display Module<br/>ResultsPage"]
        C["Visualization Analysis Module<br/>VizDisplayPage + 12 Viz Components"]
        D["Model Comparison Module<br/>ComparePage"]
    end

    subgraph APIGateway["API Gateway Module"]
        E["Flask Router<br/>RESTful API /api/v1/*"]
    end

    subgraph BackendModule["Backend Service Module"]
        F["Model Inference Module<br/>main_model.py"]
        G["Task Scheduling Module<br/>Celery tasks.py"]
        H["Data Cache Module<br/>Redis"]
        I["Structure Prediction Module<br/>LinearFold"]
        J["Data Processing Module<br/>human.py / One-hot Encoding"]
        K["Attribution Analysis Module<br/>Captum IG"]
    end

    A -->|POST Sequence| E
    B -->|GET Results| E
    C -->|GET Visualization Data| E
    D -->|GET Comparison Data| E

    E -->|Synchronous Inference| F
    E -->|Async Task| G
    E -->|Cache Read/Write| H
    E -->|Attribution Computation| K

    G -->|Model Call| F
    G -->|Structure Prediction| I
    G -->|Result Cache| H
    F -->|Data Preprocessing| J
    F -->|Graph Construction| I
```

**Detailed Module Function Description**:

1. **Sequence Input Module (WorkspacePage)**: Responsible for RNA sequence input, validation, and preprocessing. Supports both direct text input and FASTA file upload methods, with automatic sequence format validation (base character check, length check) and error handling.

2. **Result Display Module (ResultsPage)**: Responsible for displaying prediction results, including task status polling, classification probability display (bar chart), confidence information, etc. Supports result export in CSV/JSON format.

3. **Visualization Analysis Module (VizDisplayPage)**: Contains 12 visualization components providing deep model analysis and data exploration functions. Uses the VizLayout shared layout component for unified sidebar navigation and content area layout.

4. **Model Comparison Module (ComparePage)**: Supports comparative analysis of multi-model performance metrics, including visualization comparison of accuracy, precision, recall, F1 scores, and other evaluation metrics.

5. **Model Inference Module (main_model.py)**: Responsible for RGCNFormer model loading and inference computation. Supports both PyTorch native model and ONNX format model loading methods, with model paths specified through configuration files.

6. **Task Scheduling Module (Celery tasks.py)**: Responsible for async task scheduling and execution, defining two core tasks: `run_prediction_task` (single inference) and `process_sequence_in_batch` (batch inference subtask).

7. **Data Cache Module (Redis)**: Responsible for prediction result caching, task status storage, and WeChat user session management. Implements cache reuse for identical sequences through SHA256 hash keys.

8. **Structure Prediction Module (LinearFold)**: Responsible for RNA secondary structure prediction, providing dot-bracket format structure information for graph construction.

9. **Data Processing Module (human.py)**: Responsible for data preprocessing, including one-hot encoding, sequence length standardization (padding/truncation to 1001nt), and graph edge index construction.

10. **Attribution Analysis Module (Captum IG)**: Responsible for integrated gradients attribution analysis, computing each base site's contribution to prediction results.

### 2.4 Data Flow Design

Data Flow Diagrams (DFD) are important tools for describing data flow processes within a system. This section uses three levels of data flow diagrams to progressively show data processing within the system from macro to micro perspectives.

#### 2.4.1 Top-Level Data Flow Diagram

As shown in Figure 4, the top-level data flow diagram (also known as context diagram) illustrates data interactions between the system and external entities. The system receives RNA sequences submitted by users and returns prediction results and visualization data after internal processing.

**Figure 4 System Top-Level Data Flow Diagram**

```mermaid
graph LR
    User((User)) -->|RNA Sequence<br/>Parameter Config| System["RNA Sequence Modification<br/>Detection Visualization System"]
    System -->|Prediction Results<br/>Visualization Data<br/>Analysis Report| User
    Admin((Admin)) -->|Model Config<br/>System Parameters| System
    System -->|System Status<br/>Runtime Logs| Admin
```

#### 2.4.2 First-Level Data Flow Diagram

As shown in Figure 5, the first-level data flow diagram decomposes the system into 5 main processing processes, showing the main data processing flow within the system.

**Figure 5 System First-Level Data Flow Diagram**

```mermaid
graph LR
    User((User)) -->|RNA Sequence| P1["Sequence Input & Validation"]
    P1 -->|Validated Sequence| P2["Data Preprocessing"]
    P2 -->|One-hot Encoding<br/>Graph Structure| P3["Model Inference"]
    P3 -->|Prediction Results| P4["Result Generation & Caching"]
    P4 -->|Visualization Data| P5["Visualization Rendering"]
    P5 -->|Graphical Results| User

    P2 -->|RNA Sequence| P6["Secondary Structure Prediction"]
    P6 -->|dot-bracket Structure| P2
```

#### 2.4.3 Second-Level Data Flow Diagram (Core Inference Flow)

As shown in Figure 6, the second-level data flow diagram details the data processing specifics of the core inference flow, including the complete data transformation process from sequence input to result return.

**Figure 6 Core Inference Flow Data Flow Diagram**

```mermaid
graph TD
    A["RNA Sequence String"] --> B["One-hot Encoding<br/>Output: N×4 Matrix"]
    B --> C{"Sequence Length Check"}
    C -->|< 1001nt| D["Symmetric Padding<br/>Output: 1001×4"]
    C -->|≥ 1001nt| E["Middle Truncation<br/>Output: 1001×4"]
    D --> F["LinearFold Secondary Structure Prediction"]
    E --> F
    F --> G["dot-bracket String"]
    G --> H["Graph Construction<br/>build_edge_index"]
    H --> I["Base Pairing Edges<br/>A-U, G-C, G-U"]
    H --> J["Sequential Adjacency Edges<br/>i, i+1"]
    I --> K["edge_index 2×E"]
    J --> K
    K --> L["RGCNFormer Forward Inference"]
    D --> L
    E --> L
    L --> M["logits + probabilities"]
    L --> N["attention weight matrix"]
    L --> O["secondary structure data"]
    M --> P["Result Serialization to JSON"]
    N --> P
    O --> P
    P --> Q["Write to Redis Cache<br/>SET + TTL"]
    Q --> R["Return JSON Response to Frontend"]
```

**Detailed Data Flow Description**:

1. **One-hot Encoding**: Converts RNA sequence strings to (N, 4) numerical matrices. Each base is represented by a 4-dimensional one-hot vector: A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], U/T=[0,0,0,1]. For N bases (unknown bases), a uniform distribution vector [0.25,0.25,0.25,0.25] is used.

2. **Length Standardization**: Unifies sequences to 1001nt length. Sequences shorter than 1001nt are symmetrically padded with N bases at both left and right ends; sequences longer than 1001nt are truncated from the middle position to 1001nt. This design ensures consistent model input dimensions.

3. **LinearFold Secondary Structure Prediction**: Calls the LinearFold tool to predict RNA secondary structure, outputting dot-bracket format strings. LinearFold uses an approximate algorithm with linear time complexity, capable of completing structure prediction for long sequences in seconds.

4. **Graph Construction**: Builds graph structures based on secondary structure information. Edge index contains two types of edges: base pairing edges (from pairing relationships in secondary structure, including Watson-Crick pairing A-U, G-C and Wobble pairing G-U) and sequential adjacency edges (phosphodiester bond connections between i and i+1). Edge index format is (2, E), where E is the total number of edges.

5. **RGCNFormer Forward Inference**: Feeds encoded sequence features and graph structure into the RGCNFormer model. The model proceeds through three stages: ParallelCNNBlock extracting multi-scale sequence features, GCNBlock fusing graph structure information, and ClassQueryHead generating 12-class prediction probabilities, outputting logits, softmax probabilities, and attention weight matrices.

6. **Result Serialization and Caching**: Converts model outputs to JSON format, including 12-class classification probabilities, attention matrices, secondary structure data, etc. Results are stored in Redis cache with configurable TTL (default 24 hours), enabling cache reuse for identical sequences.

### 2.5 API Design Overview

The system adopts RESTful API design specifications, following these design principles:

1. **Resource-oriented**: Uses nouns to represent resources (e.g., `results`, `model-architecture`), with HTTP methods representing operations (GET for queries, POST for submissions).
2. **Unified response format**: All interfaces return unified JSON format containing three fields: `code` (status code), `message` (status description), and `data` (data payload).
3. **Version management**: All interface paths contain version numbers (`/api/v1/`), facilitating subsequent upgrades and backward compatibility.
4. **Cross-origin support**: Through Flask-CORS middleware for cross-origin request support, configuring allowed origins, methods, and headers.
5. **Error handling**: Unified error response format containing detailed error codes and error descriptions, facilitating frontend error handling.

The WeChat Mini Program uses dedicated interfaces (`/api/v1/wx-*`), with interface design considering the Mini Program's special limitations such as request frequency limits (maximum 5 per second), data size limits (single request not exceeding 1MB), etc.

---

## Part III: Detailed System Design

Building upon the overall system design, this chapter elaborates on the detailed design schemes for both frontend and backend. The frontend section covers Web frontend architecture design (component hierarchy, route design, state management, visualization component design) and WeChat Mini Program frontend design (page structure, data flow, functional differences from Web). The backend section covers API interface detailed design (request parameters and response formats for 12 core endpoints), core processing flows (inference sequence diagrams), model inference module design (RGCNFormer three-stage network structure), data processing module design (one-hot encoding and graph construction), cache and task scheduling design (Redis caching strategy and Celery task queues), and data storage design (Redis data model ER diagrams). The detailed design transforms the overall design scheme into implementable technical specifications, with the goal of providing developers with clear, executable technical guidance to ensure consistency between system implementation and design specifications.

### 3.1 Frontend Detailed Design

#### 3.1.1 Web Frontend Architecture

##### 3.1.1.1 Technology Stack

The Web frontend uses a modern frontend technology stack, with each technology component selected after thorough technical research and performance testing:

- **Framework**: React 19 + TypeScript 5.9, adopting functional components and Hooks patterns for declarative UI development.
- **Build Tool**: Vite 7.x, ESM-based fast build tool supporting millisecond-level Hot Module Replacement (HMR).
- **UI Component Library**: Ant Design 6.x, providing rich high-quality UI components following unified design specifications.
- **Visualization Libraries**: ECharts 6 (statistical charts), AntV G6 (graph structure visualization), D3.js (data-driven visualization), ReactFlow (flow diagrams), Three.js (3D rendering).
- **State Management**: React Context (global state) + React Query v5 (server-side state management).
- **Route Management**: react-router-dom v7 (HTML5 History API).
- **Internationalization**: Custom LanguageContext supporting Chinese (zh.ts) and English (en.ts).

##### 3.1.1.2 Component Hierarchy

As shown in Figure 7, the Web frontend adopts a component-based architecture with clear component hierarchy and well-defined responsibilities. The top-level component App.tsx handles route configuration and global state initialization, with each page component responsible for specific functional domain UI rendering and interaction logic.

**Figure 7 Web Frontend Component Hierarchy Diagram**

```mermaid
graph TD
    App["App.tsx<br/>Route Config + Global State Init"]
    
    App --> WP["WorkspacePage.tsx<br/>Workspace - Sequence Input"]
    App --> MP["MainPage.tsx<br/>System Entry/Navigation (/legacy)"]
    App --> RP["ResultsPage.tsx<br/>Results Overview"]
    App --> VDP["VizDisplayPage.tsx<br/>Visualization Display Container"]
    App --> CP["ComparePage.tsx<br/>Model Comparison"]
    
    WP --> W1["SequenceInput Component<br/>Sequence Input Box"]
    WP --> W2["FileUpload Component<br/>File Upload"]
    WP --> W3["ParamConfig Component<br/>Parameter Configuration"]

    VDP --> V1["ClassificationViz<br/>12-Class Classification Results"]
    VDP --> V2["LocalizationViz<br/>Modification Site Localization"]
    VDP --> V3["AttentionViz<br/>Attention Weights"]
    VDP --> V4["AttentionComparisonViz<br/>Attention Comparison"]
    VDP --> V5["AttentionDistributionViz<br/>Attention Distribution"]
    VDP --> V6["GcnViz<br/>GCN Graph Structure"]
    VDP --> V7["TargetGcnViz<br/>Target Node GCN"]
    VDP --> V8["IntegratedGradientsViz<br/>IG Attribution"]
    VDP --> V9["UMapViz<br/>UMAP Dimensionality Reduction"]
    VDP --> V10["ModelViz<br/>Model Structure"]
    VDP --> V11["RgcnformerHeatmap<br/>Heatmap"]
    VDP --> V12["LocComparisonViz<br/>Site Comparison"]

    V1 --- VL["VizLayout.tsx<br/>Shared Visualization Layout<br/>Sidebar Nav + Content Area"]
    V3 --- VL
    V6 --- VL
    V8 --- VL
    V10 --- VL
```

**Detailed Component Responsibilities**:

1. **App.tsx**: Application root component responsible for route configuration (10 routes), QueryClientProvider (React Query) initialization, LanguageContextProvider (internationalization) initialization, and global state management.

2. **WorkspacePage.tsx**: Workspace page, serving as the system's main entry point. Responsible for RNA sequence input (direct text input), validation (base character check, length check), and submission (calling `/api/v1/submit-task` interface). Contains sequence input box, file upload component, and parameter configuration form sub-components.

3. **ResultsPage.tsx**: Results overview page, polling prediction results through React Query's `useQuery` hook. Displays task status (processing/completed/failed), 12-class classification probability bar charts, and basic result information.

4. **VizDisplayPage.tsx**: Visualization display container, dynamically loading corresponding visualization components based on URL path parameters. Uses VizLayout shared layout component for unified sidebar navigation and content area layout.

5. **ComparePage.tsx**: Model comparison page, calling `/api/v1/model-comparison` interface to retrieve multi-model performance data, supporting visualization comparison of accuracy, precision, recall, F1 scores, and other metrics.

6. **VizLayout.tsx**: Shared visualization layout component providing responsive sidebar navigation (fixed sidebar on desktop, collapsible sidebar on mobile) and content area layout. All visualization components implement unified page structure through VizLayout.

#### 3.1.2 Frontend Route Design

As shown in Table 3, the system uses react-router-dom v7 for route management, defining 10 routes covering all system functional pages.

**Table 3 Frontend Route Table**

| Path | Page Component | Layout | Function Description |
|------|---------------|--------|---------------------|
| `/` | WorkspacePage | Independent layout | System main entry, RNA sequence input and submission |
| `/legacy` | MainPage | Independent layout | Legacy entry compatibility, system function navigation |
| `/results/:jobId` | ResultsPage | Independent layout | Prediction results overview, task status polling |
| `/viz-display` | VizDisplayPage | Independent layout | Visualization display container |
| `/classification` | ClassificationViz | VizLayout | 12-class modification classification probability display |
| `/attention` | AttentionViz | VizLayout | Multi-head attention weight heatmap |
| `/gcn` | GcnViz | VizLayout | RNA secondary structure graph visualization |
| `/target-gcn` | TargetGcnViz | VizLayout | Specific node GCN message passing |
| `/integrated-gradients` | IntegratedGradientsViz | VizLayout | Integrated gradients attribution analysis |
| `/model-viz` | ModelViz | VizLayout | Hierarchical model architecture display |
| `/compare` | ComparePage | Independent layout | Multi-model performance comparison |

**Route Design Notes**:

- **Route grouping**: The 10 routes are divided into two groups— independent layout routes (WorkspacePage, MainPage, ResultsPage, VizDisplayPage, ComparePage) and VizLayout shared layout routes (6 visualization component pages). VizLayout provides unified sidebar navigation, allowing users to quickly switch between different visualization components.

- **Route parameters**: `:jobId` is the unique identifier for prediction tasks (SHA256 hash value), used to query corresponding prediction results.

- **BrowserRouter configuration**: Uses HTML5 History API, configured with `basename="/rgcnformer"` to support sub-path deployment.

#### 3.1.3 State Management Design

The system adopts a dual-layer state management strategy using React Context and React Query, as shown in Figure 8. React Context manages global application state (e.g., language settings, user preferences), while React Query manages server-side state (e.g., API request results, cache data, polling status).

**Figure 8 Frontend State Transition Diagram**

```mermaid
stateDiagram-v2
    [*] --> Idle: Initial State
    Idle --> Loading: Trigger Request<br/>(useQuery/useMutation)
    Loading --> Success: Request Success<br/>(200 OK)
    Loading --> Error: Request Failed<br/>(Network Error/Server Error)
    Success --> Idle: Data Expired/Manual Invalidation
    Error --> Loading: Auto Retry<br/>(retry config)
    Error --> Idle: Manual Reset

    state Loading {
        [*] --> Fetching: Send HTTP Request
        Fetching --> Polling: Poll Task Status<br/>(every 2 seconds)
        Polling --> Fetching: Receive Intermediate Status
    }
```

**Detailed State Management Strategy**:

1. **React Context**: Used for managing global application state, including:
   - `LanguageContext`: Manages current language settings (Chinese/English), providing `t(key)` translation function.
   - Global theme configuration: Manages dark/light theme switching.

2. **React Query (@tanstack/react-query v5)**: Used for managing server-side state, providing the following core functions:
   - **Request caching**: `useQuery` hook caches requested data by default. When users access the same data again, cached results are returned first while background data refresh occurs (stale-while-revalidate strategy).
   - **Auto-polling**: For asynchronous inference tasks, configuring `refetchInterval` to implement automatic polling until task status changes to completed or failed.
   - **Error retry**: Configuring `retry` parameters to implement automatic retry after request failures, supporting exponential backoff strategy.
   - **Optimistic updates**: `useMutation` hook supports optimistic updates, updating local cache while waiting for server response.

3. **Request state transitions**:
   - **Idle**: Initial state, no active requests.
   - **Loading**: Request in progress, displaying loading indicator.
   - **Success**: Request successful, displaying data.
   - **Error**: Request failed, displaying error information and retry button.

#### 3.1.4 Visualization Component Design

As shown in Table 4, each visualization component's data source interface, visualization library selection, and interaction methods are carefully designed to ensure optimal display effects and interaction experience.

**Table 4 Visualization Component Data Source and Interaction Design Table**

| Component | Data Source Interface | Visualization Library | Interaction Method | Main Function |
|-----------|----------------------|----------------------|-------------------|---------------|
| ClassificationViz | `/api/v1/results/:jobId` | ECharts | Hover for details, click to expand | 12-class modification probability bar/radar chart |
| LocalizationViz | `/api/v1/results/:jobId` | ECharts | Site click highlight | Modification site distribution annotation on sequence |
| AttentionViz | `/api/v1/results/:jobId` | ECharts | Heatmap zoom, drag | Multi-head attention weight heatmap display |
| AttentionComparisonViz | `/api/v1/results/:jobId` | ECharts | Comparison toggle | Attention comparison across modification types |
| AttentionDistributionViz | `/api/v1/results/:jobId` | ECharts | Distribution parameter adjustment | Attention weight statistical distribution |
| GcnViz | `/api/v1/results/:jobId` | AntV G6 | Node drag, zoom, hover | RNA secondary structure force-directed graph |
| TargetGcnViz | `/api/v1/visualize-gcn-aggregation` | ReactFlow | Node selection, expand | GCN neighborhood aggregation message passing flow |
| IntegratedGradientsViz | `/api/v1/integrated-gradients` | D3.js + G6 | Attribution value exploration, region selection | Base-level integrated gradients attribution analysis |
| UMapViz | `/api/v1/umap` | ECharts | Scatter zoom, region selection | UMAP dimensionality reduction scatter plot |
| ModelViz | `/api/v1/model-architecture` | ReactFlow | Hierarchy expand, node details | RGCNFormer hierarchical architecture diagram |
| RgcnformerHeatmap | `/api/v1/results/:jobId` | ECharts | Heatmap zoom | Modification site prediction heatmap |
| LocComparisonViz | `/api/v1/results/:jobId` | ECharts | Site comparison toggle | Different modification site prediction comparison |

**Detailed Visualization Component Interaction Design**:

1. **ClassificationViz**: Displays 12-class modification prediction probabilities in bar chart, with x-axis as modification types (Am, Atol, Cm, etc.) and y-axis as prediction probability values (0~1). Supports hovering for detailed values and confidence intervals, with radar chart view toggle.

2. **LocalizationViz**: Highlights predicted modification sites on the sequence, with different colors distinguishing modification types (e.g., m6A in red, m5C in blue). Supports clicking for site details, including the 12-class modification probability distribution at that site.

3. **AttentionViz**: Displays multi-head attention weights in heatmap format, with x-axis as sequence positions (0~1000) and y-axis as attention heads (1~8), with color intensity representing attention weight magnitude. Supports zoom and drag operations, with specific attention head selection.

4. **GcnViz**: Displays RNA secondary structure as force-directed graph, with nodes representing bases (A, C, G, U in different colors) and edges representing base pairing relationships (solid lines for Watson-Crick pairing, dashed lines for Wobble pairing) and sequential adjacency relationships. Supports node drag layout adjustment, mouse wheel zoom, and node hover detail viewing.

5. **IntegratedGradientsViz**: Displays integrated gradients attribution results, with x-axis as sequence positions and y-axis as attribution values. Highlights base sites with highest contribution to prediction results (positive attribution in red, negative attribution in blue), supporting region selection and zoom exploration.

6. **UMapViz**: Displays UMAP dimensionality reduction results in scatter plot, with each scatter point representing an RNA sequence feature embedding, and different colors indicating different modification types. Supports zoom, region selection, and scatter point hover for sequence information.

7. **ModelViz**: Displays RGCNFormer model network architecture in hierarchical flow diagram, sequentially showing ParallelCNNBlock, GCNBlock, and ClassQueryHead three core modules from input to output layers. Supports hierarchy expansion for detail viewing, with node hover displaying parameter information.

#### 3.1.5 WeChat Mini Program Frontend Design

##### 3.1.5.1 Page Structure

The WeChat Mini Program uses native framework development based on the glass-easel component framework, containing 4 pages as shown in Table 5.

**Table 5 WeChat Mini Program Page Structure**

| Page | Path | Function Description | Core Components |
|------|------|---------------------|-----------------|
| Home | `pages/index/index` | RNA sequence input, supports up to 5 batch submissions | Sequence input box, submit button, login button |
| Results | `pages/results/results` | 12-class modification prediction results display | Classification result list, attention weight bar |
| WebView | `pages/webview/index` | Embedded web-view for redirecting to Web platform for 3D visualization | Native web-view component |
| Logs | `pages/logs/logs` | WeChat Mini Program log viewing | Log list |

**Mini Program Data Flow**:

1. User enters 1~5 RNA sequences on the home page (each ≥51nt, only ACGUTN characters).
2. Clicks submit button, calls WeChat login interface to obtain openid, then POSTs to `/api/v1/wx-submit-task`.
3. System returns batch_job_id, Mini Program polls `/api/v1/wx-task-progress/{jobId}` every 2 seconds to query progress.
4. After task completion, redirects to results page to display 12-class modification prediction probabilities.
5. User can choose to redirect to web-view page for full visualization analysis on the Web platform.

##### 3.1.5.2 Functional Differences from Web Platform

As shown in Table 6, the Mini Program and Web platform have clear functional positioning differences.

**Table 6 Mini Program vs Web Platform Feature Comparison**

| Feature Module | Mini Program | Web Platform | Difference |
|---------------|-------------|-------------|------------|
| Sequence Input | Supported (up to 5) | Supported (unlimited) | Mini Program limits batch size for resource control |
| Batch Submission | Supported | Supported | Both support async batch inference |
| Prediction Progress | Polling view (2s interval) | React Query auto-polling | Different polling mechanisms |
| Classification Results | Supported (list display) | Supported (chart display) | Web has richer charts |
| Attention Visualization | Not supported | Supported (ECharts heatmap) | Mini Program redirects to Web |
| GCN Graph Visualization | Not supported | Supported (AntV G6 force-directed) | Mini Program redirects to Web |
| UMAP Visualization | Not supported | Supported (ECharts scatter) | Mini Program redirects to Web |
| IG Attribution Analysis | Not supported | Supported (D3.js attribution) | Mini Program redirects to Web |
| Model Comparison | Not supported | Supported (ComparePage) | Mini Program redirects to Web |
| Internationalization | Not supported | Supported (Chinese/English) | Mini Program Chinese only |
| 3D Visualization | Via web-view redirect | Native support (Three.js) | Mini Program depends on Web |

The Mini Program is positioned as a **lightweight query tool** with core functions being sequence submission, progress monitoring, and result viewing; complex visualization functions are implemented through embedded web-view component redirection to the Web platform.

### 3.2 Backend Detailed Design

#### 3.2.1 API Interface Detailed Design

##### 3.2.1.1 Core API Endpoint List

As shown in Table 7, the system defines a total of 12 core API endpoints covering task submission, result querying, visualization data retrieval, model management, and system monitoring functions.

**Table 7 Core API Endpoint List**

| ID | Endpoint Name | Method | Path | Request Parameters | Response Format | Description |
|----|--------------|--------|------|-------------------|----------------|-------------|
| 1 | Submit Task | POST | `/api/v1/submit-task` | `{rnaSequence, targetClassId?, topK?}` | `{jobId, status}` | Single async inference, SHA256 cache key |
| 2 | Batch Submit | POST | `/api/v1/wx-submit-task` | `{sequences: [seq1..seq5]}` | `{batch_job_id}` | WeChat batch, UUID identifier |
| 3 | Query Results | GET | `/api/v1/results/:jobId` | — | Full result JSON | Polling retrieval, supports intermediate status |
| 4 | Batch Progress | GET | `/api/v1/wx-task-progress/:jobId` | — | `{status, progress, results}` | WeChat progress query |
| 5 | IG Attribution | POST | `/api/v1/integrated-gradients` | `{rnaSequence, targetClassId}` | `{attributions, nodes, edges}` | Captum integrated gradients computation |
| 6 | GCN Aggregation | POST | `/api/v1/visualize-gcn-aggregation` | `{rnaSequence, targetNodeIdx}` | `{nodes, edges, aggregationData}` | GCN message passing visualization |
| 7 | Model Architecture | GET | `/api/v1/model-architecture` | — | Hierarchical JSON tree | PyTorch model structure |
| 8 | Model Compute Graph | GET | `/api/v1/model-graph` | — | `{nodes, edges}` | ONNX compute graph data |
| 9 | Model Comparison | GET | `/api/v1/model-comparison` | — | `{models, metrics}` | Multi-model performance comparison |
| 10 | UMAP Reduction | GET | `/api/v1/umap` | — | `{points, labels}` | Pre-computed UMAP embeddings |
| 11 | Sample Sequence | GET | `/api/v1/sample-sequence` | — | `{sequence, name}` | Random RNA example sequence |
| 12 | Health Check | GET | `/api/health` | — | `{status, model_loaded, device}` | System health status |

##### 3.2.1.2 Detailed Interface Description

**1. Submit Task Interface (POST /api/v1/submit-task)**

This is the system's most core interface, responsible for receiving user RNA sequence prediction requests. Interface processing flow:

Request parameters:
- `rnaSequence` (required, string): RNA sequence string, only containing A, C, G, U, T, N characters, length ≥51nt.
- `targetClassId` (optional, int): Target modification type ID (0~11), used for specific modification attribution analysis.
- `topK` (optional, int): Return Top-K prediction results, default is 12.

Response format:
```json
{
  "jobId": "a1b2c3d4e5f6...",
  "status": "pending"
}
```

Processing logic:
1. Validate sequence format (only legal base characters, length ≥51nt).
2. Compute sequence SHA256 hash value as jobId.
3. Query Redis cache (`task:{sha256}`); if hit, directly return cached result (status: "completed").
4. If cache misses, asynchronously submit inference task through Celery (`run_prediction_task.apply_async`).
5. Return jobId and status "pending", HTTP status code 202.

**2. Query Results Interface (GET /api/v1/results/:jobId)**

Response format (when task completes):
```json
{
  "status": "completed",
  "classification": {
    "m6A": 0.85,
    "m5C": 0.12,
    "Ψ": 0.03,
    "ac4C": 0.01,
    ...
  },
  "attention": [[0.1, 0.2, ...], ...],
  "probabilities": [0.85, 0.12, 0.03, ...],
  "structure": "..((..))..",
  "nodes": [{"id": 0, "label": "A", "x": 0.5, "y": 0.3}, ...],
  "edges": [{"source": 0, "target": 1, "type": "adjacent"}, ...]
}
```

**3. Health Check Interface (GET /api/health)**

Response format:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "checkpoint_path": "/data/models/mrmodn_best.pth"
}
```

#### 3.2.2 Core Processing Flow

##### 3.2.2.1 Inference Flow Sequence Diagram

As shown in Figure 9, the inference flow sequence diagram details the complete interaction process from user sequence submission to result retrieval, including both cache hit and cache miss scenarios. This sequence diagram covers interactions among six participants: Frontend, Flask, Redis, Celery, LinearFold, and RGCNFormer.

**Figure 9 Inference Flow Sequence Diagram**

```mermaid
sequenceDiagram
    participant Frontend as Frontend<br/>(React/Mini Program)
    participant Flask as Flask API
    participant Redis as Redis Cache
    participant Celery as Celery Worker
    participant LF as LinearFold
    participant Model as RGCNFormer

    Frontend->>Flask: POST /api/v1/submit-task<br/>{rnaSequence}
    Flask->>Flask: Compute SHA256(jobId)
    Flask->>Redis: GET task:{sha256}

    alt Cache Hit
        Redis-->>Flask: Return cached result
        Flask-->>Frontend: 200 OK (full result JSON)
    else Cache Miss
        Flask->>Celery: apply_async(run_prediction_task)
        Flask-->>Frontend: 202 Accepted {jobId, status: "pending"}
        
        Note over Frontend: Frontend begins polling

        loop Poll every 2 seconds
            Frontend->>Flask: GET /api/v1/results/:jobId
            Flask->>Redis: Query task status
            Redis-->>Flask: status: "processing"
            Flask-->>Frontend: {status: "processing"}
        end

        Celery->>LF: run_linearfold([sequence])
        LF-->>Celery: dot-bracket secondary structure
        Celery->>Celery: build_edge_index_from_structure()
        Celery->>Model: model.forward(x, edge_index, batch)
        Model-->>Celery: {logits, probabilities, attention}
        Celery->>Redis: SET task:{sha256} (result JSON, TTL)

        Frontend->>Flask: GET /api/v1/results/:jobId
        Flask->>Redis: GET task:{sha256}
        Redis-->>Flask: Return cached result
        Flask-->>Frontend: 200 OK (full result JSON)
    end
```

**Detailed Flow Description**:

1. **Request Submission Phase**: Frontend sends POST request to Flask with RNA sequence string. After validating sequence format, Flask computes SHA256 hash value as task identifier.

2. **Cache Query Phase**: Flask queries Redis cache with key `task:{sha256}`. If cache hits (same sequence has been predicted before), cached results are directly returned, avoiding redundant computation.

3. **Task Dispatch Phase**: If cache misses, Flask asynchronously submits inference task through Celery's `apply_async` method, immediately returning HTTP 202 Accepted status code and task ID.

4. **Inference Execution Phase**: Celery Worker executes the inference task: first calling LinearFold to predict RNA secondary structure, then constructing graph edge indices (base pairing edges + sequential adjacency edges), and finally executing RGCNFormer forward inference to obtain logits, probabilities, and attention weights.

5. **Result Caching Phase**: After inference completion, Celery stores result JSON to Redis cache, setting TTL (default 24 hours).

6. **Result Retrieval Phase**: Frontend polls for results, Redis returns complete result JSON, and frontend performs visualization rendering.

#### 3.2.3 Model Inference Module Design

##### 3.2.3.1 RGCNFormer Model Structure

As shown in Figure 10, the RGCNFormer model adopts a three-stage pipeline architecture: Multi-scale CNN Feature Extraction → Graph Convolution Structure Fusion → Class-Query Attention Classification. This design achieves progressive feature abstraction from local sequence patterns to global structural information to classification decisions.

**Figure 10 RGCNFormer Model Structure Diagram**

```mermaid
graph TD
    Input["Input: RNA Sequence<br/>(1001, 4) One-hot Encoding"]

    subgraph Stage1["Stage 1: ParallelCNNBlock (M2D Multi-scale CNN Module)"]
        Conv1["Conv1d(kernel=1)<br/>Single-base Features<br/>Output: 16 channels"]
        Conv3["Conv1d(kernel=3)<br/>3-mer Patterns<br/>Output: 16 channels"]
        Conv5["Conv1d(kernel=5)<br/>5-mer Patterns<br/>Output: 16 channels"]
        Conv7["Conv1d(kernel=7)<br/>7-mer Patterns<br/>Output: 16 channels"]
        Concat["Feature Concatenation<br/>Output: 64 channels"]
        LN1["LayerNorm + ReLU<br/>+ Dropout(0.1)"]
    end

    subgraph Stage2["Stage 2: GCNBlock (Graph Convolution Module)"]
        GCN1["GCNConv Layer 1<br/>64 → 128<br/>+ Residual Connection"]
        GCN2["GCNConv Layer 2<br/>128 → 128<br/>+ Residual Connection"]
        GCN3["GCNConv Layer 3<br/>128 → 128<br/>+ Residual Connection"]
        LN2["LayerNorm + ReLU<br/>+ Dropout(0.3)"]
    end

    subgraph Stage3["Stage 3: ClassQueryHead (Class-Query Attention Module)"]
        CQ["12 Learnable Class Query Vectors<br/>(12, 128)"]
        MHA["Multi-Head Attention<br/>heads=8"]
        Softmax["Softmax Classification<br/>Output: 12-class probabilities"]
    end

    Input --> Conv1 & Conv3 & Conv5 & Conv7
    Conv1 & Conv3 & Conv5 & Conv7 --> Concat --> LN1
    LN1 --> GCN1 --> GCN2 --> GCN3 --> LN2
    LN2 --> MHA
    CQ --> MHA
    MHA --> Softmax

    EdgeIndex["edge_index<br/>(Base Pairing Edges + Sequential Adjacency Edges)"] --> GCN1
    EdgeIndex --> GCN2
    EdgeIndex --> GCN3

    Softmax --> Output["Output:<br/>12-class Modification Probabilities<br/>Attention Weight Matrix<br/>Secondary Structure Data"]
```

**Detailed Module Description**:

**1. ParallelCNNBlock (Multi-scale CNN Module)**

ParallelCNNBlock adopts a multi-scale parallel convolution strategy, using four 1D convolutional kernels with kernel sizes of 1, 3, 5, and 7 to extract sequence patterns of different granularities in parallel:

- **kernel=1**: Captures single-base level features, focusing on base type information at each position.
- **kernel=3**: Captures 3-mer (trinucleotide) patterns, focusing on combined features of three adjacent bases.
- **kernel=5**: Captures 5-mer patterns, focusing on longer local sequence motifs.
- **kernel=7**: Captures 7-mer patterns, focusing on broader sequence context.

Each convolution branch outputs `hidden_dim // 4 = 64 // 4 = 16` channels, with four branch outputs concatenated via `torch.cat` to 64 channels. The concatenated features undergo LayerNorm normalization, ReLU activation, and Dropout(0.1) regularization.

**Input dimension**: (batch_size × 1001 × 4) → transposed to (batch_size × 4 × 1001)
**Output dimension**: (total_nodes × 64), where total_nodes = batch_size × 1001

**2. GCNBlock (Graph Convolution Module)**

GCNBlock performs graph convolution computation based on RNA secondary structure information, progressively fusing graph structure information through 3 GCNConv layers. Each GCNConv layer's computation process:

$$h_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \frac{1}{c_{ij}} W^{(l)} h_j^{(l)} + b^{(l)}\right)$$

Where $h_i^{(l)}$ is the feature vector of node $i$ at layer $l$, $\mathcal{N}(i)$ is the neighbor set of node $i$, $c_{ij}$ is the normalization coefficient, and $W^{(l)}$ is the learnable weight matrix.

- **Input projection**: 64 dimensions → 128 dimensions (when input dimension doesn't match hidden dimension).
- **Hidden layers**: 3 GCNConv layers, hidden dimension 128, each followed by LayerNorm + ReLU + Dropout(0.3).
- **Residual connections**: Each GCNConv layer's output is residual-connected with its input, alleviating gradient vanishing in deep graph networks.

**Input dimension**: (total_nodes × 64) + edge_index(2, E)
**Output dimension**: (total_nodes × 128)

**3. ClassQueryHead (Class-Query Attention Module)**

ClassQueryHead adopts a class-query attention mechanism, defining 12 learnable class query vectors $Q \in \mathbb{R}^{12 \times 128}$ corresponding to 12 RNA modification types. Through multi-head attention computation (8 attention heads), class query vectors interact with node features:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where $K$ and $V$ come from GCNBlock's output node features, and $Q$ is the 12 class query vectors. The attention output passes through fully connected layers and Softmax activation to generate 12-class modification prediction probabilities.

**Input dimension**: Class queries (12 × 128) + Node features (total_nodes × 128)
**Output dimension**: (batch_size × 12), softmax probabilities for 12 modification classes

##### 3.2.3.2 Model Variant Comparison

As shown in Table 8, the system supports multiple model variants to meet the needs of different research scenarios.

**Table 8 Model Variant Comparison Table**

| Variant | File | Architecture Features | Application Scenario | Parameters |
|---------|------|----------------------|---------------------|------------|
| mRModN | `mrmodn.py` | Main network, standard three-stage RGCNFormer | General 12-class modification prediction | ~2.1M |
| ModX | `modx.py` | Modification type ablation variant, removing specific modules | Ablation studies | Configurable |
| MultiRM | `multirm.py` | Multi-task multi-modification joint learning | Joint multi-modification prediction | ~3.5M |
| EvoRMD | `evormd_human.py` | Integrating evolutionary conservation features | Generalization ability research | ~2.8M |
| AblaModel | `abla_model.py` | Dedicated ablation study model, modules switchable | 3×3 ablation matrix | Configurable |

#### 3.2.4 Data Processing Module Design

##### 3.2.4.1 Data Processing Flow

As shown in Figure 11, the data processing flow includes four main steps: sequence encoding, length standardization, secondary structure prediction, and graph construction, forming a complete data pipeline from raw sequence strings to model-ready inputs.

**Figure 11 Data Processing Flow Diagram**

```mermaid
graph TD
    A["RNA Sequence String<br/>(e.g.: AUGCAUGCAUGC...)"] --> B["Base Character Validation<br/>Only A/C/G/U/T/N allowed"]
    B -->|Validation Passed| C["Base Mapping<br/>T→U Conversion"]
    B -->|Validation Failed| Error["Return Error Prompt"]
    C --> D["One-hot Encoding<br/>A=[1,0,0,0]<br/>C=[0,1,0,0]<br/>G=[0,0,1,0]<br/>U=[0,0,0,1]<br/>N=[0.25,0.25,0.25,0.25]<br/>Output: (N, 4)"]

    D --> E{"Length Check"}
    E -->|N < 1001| F["Symmetric Padding<br/>N bases added to both ends<br/>Output: (1001, 4)"]
    E -->|N = 1001| G["Direct Use<br/>Output: (1001, 4)"]
    E -->|N > 1001| H["Middle Truncation<br/>Output: (1001, 4)"]

    F --> I["LinearFold<br/>Secondary Structure Prediction"]
    G --> I
    H --> I

    I --> J["dot-bracket String<br/>(e.g.: ..((..))..)"]
    J --> K["Graph Construction<br/>build_edge_index"]

    K --> L["Base Pairing Edges<br/>A-U, G-C, G-U pairings"]
    K --> M["Sequential Adjacency Edges<br/>i to i+1 connections"]
    L --> N["edge_index (2, E)<br/>E = pairing edges + adjacency edges"]
    M --> N

    N --> O["Build PyG Data Object<br/>x=(1001,4), edge_index=(2,E)<br/>+ attention masks"]
```

**Detailed Processing Steps**:

1. **Base Character Validation**: Checks that input sequences contain only six legal characters: A, C, G, U, T, N, rejecting inputs containing other characters.

2. **Base Mapping**: Automatically converts T bases to U bases (RNA uses U instead of T).

3. **One-hot Encoding**: Converts each base to a 4-dimensional one-hot vector. For N bases, a uniform distribution vector [0.25, 0.25, 0.25, 0.25] represents base uncertainty.

4. **Length Standardization**: Unifies sequences to 1001nt length. This length is chosen based on training data statistical distribution, covering modification sites of most RNA sequences.
   - **Padding**: Sequences shorter than 1001nt are symmetrically padded with N bases at both left and right ends, ensuring center region sequence information doesn't shift.
   - **Truncation**: Sequences longer than 1001nt are truncated from the middle position to 1001nt, preserving the center region.

5. **LinearFold Secondary Structure Prediction**: Calls the LinearFold C++ tool, inputting RNA sequences and outputting dot-bracket format secondary structure strings.

6. **Graph Construction**: Builds PyG graph data based on secondary structure information. Edge index contains two types of edges:
   - **Base pairing edges**: From pairing relationships in secondary structure, including Watson-Crick pairing (A-U, G-C) and Wobble pairing (G-U).
   - **Sequential adjacency edges**: Connections between each base and its front/back adjacent bases (i↔i+1), reflecting RNA chain covalent connections.
   
   Final output is a PyG Data object containing node features `x=(1001, 4)`, edge index `edge_index=(2, E)`, labels `y=(1, 12)`, site labels `y_site=(1001,)`, and various attention masks (`attn_mask_A/C/G/U` and `attn_mask_N`).

#### 3.2.5 Cache and Task Scheduling Design

##### 3.2.5.1 Redis Caching Strategy

As shown in Figure 12, the system adopts multi-level caching design with Redis, covering prediction result caching, task status caching, and user session caching.

**Figure 12 Caching Strategy Flow Diagram**

```mermaid
graph TD
    A["Request Arrives"] --> B["Compute Cache Key<br/>SHA256(sequence)"]
    B --> C["Query Redis Cache<br/>GET task:{sha256}"]

    C --> D{"Cache Hit?"}
    D -->|Yes| E["Deserialize Cache Result<br/>Return Full JSON"]
    D -->|No| F["Submit Celery Async Task<br/>run_prediction_task"]

    F --> G["Return Task ID<br/>HTTP 202 Accepted"]
    G --> H["Celery Worker Execution"]
    H --> I["LinearFold Structure Prediction"]
    I --> J["Graph Construction"]
    J --> K["RGCNFormer Inference"]
    K --> L["Serialize Result to JSON"]
    L --> M["Write to Redis Cache<br/>SET task:{sha256}<br/>EX 86400 (24h TTL)"]
    M --> N["Return Computation Result"]

    subgraph CacheKeyDesign["Cache Key Design"]
        K1["task:{sha256}<br/>Single Inference Result<br/>TTL: 24h"]
        K2["batch_job:{uuid}<br/>Batch Task Status<br/>TTL: 24h"]
        K3["wx_user:{openid}<br/>WeChat User Info<br/>TTL: 30 days"]
    end
```

**Detailed Cache Key Design**:

| Cache Key Format | Data Type | Stored Content | TTL | Description |
|-----------------|-----------|---------------|-----|-------------|
| `task:{sha256}` | String (JSON) | Full prediction result | 24 hours | Cache reuse for identical sequences |
| `batch_job:{uuid}` | Hash | Task status, progress, results | 24 hours | Batch task management |
| `wx_user:{openid}` | Hash | User info, session key | 30 days | WeChat user session |

##### 3.2.5.2 Celery Task Queue

The system defines two Celery tasks implementing async inference and batch processing:

1. **`run_prediction_task`**: Single inference task responsible for handling individual RNA sequence prediction requests. Task flow includes LinearFold structure prediction → graph construction → RGCNFormer inference → result caching.

2. **`process_sequence_in_batch`**: Batch inference subtask responsible for handling individual sequences within batch tasks. Multiple subtasks execute in parallel, updating progress through Redis Hash's `HINCRBY` command.

**Task State Transitions**:

```mermaid
stateDiagram-v2
    [*] --> PENDING: Task Created
    PENDING --> STARTED: Worker Starts Execution
    STARTED --> SUCCESS: Inference Complete
    STARTED --> FAILURE: Execution Exception
    FAILURE --> RETRY: Auto Retry
    RETRY --> STARTED: Retry Execution
    RETRY --> FAILURE: Exceeded Retry Count
    SUCCESS --> [*]: Result Written to Redis
    FAILURE --> [*]: Error Info Written to Redis
```

#### 3.2.6 Data Storage Design

##### 3.2.6.1 Redis Data Model Design

This system uses Redis as the primary data storage, without using traditional relational databases (e.g., MySQL, PostgreSQL). This design is based on the following considerations:

1. **Data characteristics**: The system primarily stores prediction results (JSON format) with flexible structures, not suitable for fixed-schema relational databases.
2. **Access patterns**: Data is accessed in key-value pair format, with read-heavy and write-light patterns. Redis's in-memory storage provides millisecond-level response times.
3. **Lifecycle**: Data has clear lifecycles (TTL), with Redis's expiration mechanism automatically cleaning expired data without additional cleanup logic.

As shown in Figure 13, the Redis data model contains three main entities and their relationships.

**Figure 13 Redis Data Model ER Diagram**

```mermaid
erDiagram
    TASK {
        string key "task:{sha256}"
        string jobId "SHA256 hash value"
        string status "completed/processing/failed"
        json classification "12-class classification probabilities"
        json attention "attention weight matrix"
        json probabilities "class probability array"
        string structure "dot-bracket secondary structure"
        json nodes "graph node data"
        json edges "graph edge data"
        int ttl "expiration time (seconds)"
    }

    BATCH_JOB {
        string key "batch_job:{uuid}"
        string batch_job_id "UUID identifier"
        string status "PENDING/PROCESSING/COMPLETED"
        int total_sequences "total sequence count"
        int completed "completed count"
        json results "per-sequence result array"
        int creation_time "creation timestamp"
        int ttl "expiration time (seconds)"
    }

    WX_USER {
        string key "wx_user:{openid}"
        string openid "WeChat unique identifier"
        string session_key "session key"
        string nickname "user nickname"
        string avatar_url "avatar URL"
        int ttl "expiration time (seconds)"
    }

    BATCH_JOB ||--o{ TASK : "contains multiple subtasks"
    WX_USER ||--o{ BATCH_JOB : "creates multiple batch tasks"
```

**Detailed Entity Relationship Description**:

1. **TASK (Single Inference Result)**: Uses sequence SHA256 hash as key, implementing cache reuse for identical sequences. When different users submit the same RNA sequence, the system directly returns cached results, avoiding redundant computation. Stored content includes prediction classification results (12-class probabilities), attention weight matrices, secondary structure data, and graph structure data.

2. **BATCH_JOB (Batch Inference Task)**: Uses UUID as key, managing batch task status and results. A batch task contains inference subtasks for multiple sequences, with a 1:N relationship to TASK. Stored using Redis Hash structure, supporting atomic progress updates (`HINCRBY` command).

3. **WX_USER (WeChat User Information)**: Uses WeChat openid as key, storing user basic information and session data. A user can create multiple batch tasks, with a 1:N relationship to BATCH_JOB.

**Detailed Redis Data Structure Description**:

| Entity | Redis Type | Key Format | Field Count | TTL |
|--------|-----------|-----------|-------------|-----|
| TASK | String (JSON) | `task:{sha256}` | 8 | 24 hours |
| BATCH_JOB | Hash | `batch_job:{uuid}` | 6 | 24 hours |
| WX_USER | Hash | `wx_user:{openid}` | 5 | 30 days |

---

## Part IV: Application Analysis of Clustering in RNA Sequence Modification Detection

This chapter provides an in-depth exploration of clustering methods in RNA sequence modification detection. Starting from fundamental concepts of clustering algorithms, it analyzes the graph clustering mechanism in the RGCNFormer model and details the implementation of UMAP embedding and clustering visualization. On this basis, clustering analysis strategies in few-shot and zero-shot scenarios are discussed separately, with ablation studies verifying each module's contribution to clustering performance. Finally, the advantages and limitations of clustering methods in RNA modification detection are summarized, with future improvement directions outlined. As an important tool in unsupervised learning, clustering analysis can discover potential patterns and structures from data, which is of great significance for understanding RNA modification distribution patterns and model internal representations. In the field of RNA modification detection, clustering analysis can not only be used for data exploration and feature visualization but also assist in evaluating model classification ability and generalization performance.

### 4.1 Overview of Clustering Methods

#### 4.1.1 Role of Unsupervised Clustering in Biological Sequence Analysis

Unsupervised clustering is a fundamental tool in data analysis, with its core goal being to partition data into several groups (clusters) such that data points within the same cluster have high similarity while data points across different clusters have low similarity. Unlike supervised learning, unsupervised clustering does not require label information and can discover potential patterns from the intrinsic structure of data.

In biological sequence analysis, clustering methods are widely applied in the following scenarios:

1. **Sequence homology analysis**: Clustering similar sequences together to discover sequence families and functional domains. For example, through clustering analysis, homologous protein sequences can be grouped into the same protein family, providing a basis for functional annotation.

2. **Structure classification**: Clustering based on structural features to discover protein or RNA structure types. RNA molecules' secondary and tertiary structures are closely related to their functions, and structural clustering can discover new structure types.

3. **Function prediction**: Inferring unknown sequence functions through clustering analysis. If a sequence of unknown function clusters with sequences of known functions, it can be inferred that the sequence has similar functions.

4. **Data dimensionality reduction and visualization**: Mapping high-dimensional features to low-dimensional space for visualization and exploration. In RNA modification detection, through UMAP dimensionality reduction and clustering visualization, distribution patterns of different modification types in feature space can be intuitively observed.

5. **Quality control**: Identifying abnormal samples and noise data in data through clustering analysis to improve data quality.

#### 4.1.2 Common Clustering Algorithms

**1. K-Means Clustering**

K-Means is the most classic partition clustering algorithm, with its basic idea being to partition $n$ data points into $K$ clusters such that each data point belongs to the cluster represented by its nearest cluster center. The algorithm updates cluster centers by iteratively optimizing the objective function (within-cluster sum of squares, i.e., SSE).

Advantages: Algorithm is simple and efficient, with time complexity $O(nKt)$ ($t$ being the number of iterations); good scalability for large datasets.

Disadvantages: Requires pre-specifying $K$ value; sensitive to initial centers, may converge to local optima; assumes clusters are convex, performing poorly on non-convex clusters.

**2. Hierarchical Clustering**

Hierarchical clustering is based on hierarchical decomposition strategies, including agglomerative (bottom-up) and divisive (top-down) approaches. Agglomerative hierarchical clustering initially treats each data point as a cluster, then progressively merges the most similar clusters until reaching a preset cluster count or satisfying stopping conditions.

Advantages: Does not require pre-specifying cluster count (can be selected through dendrogram); able to reveal hierarchical data structure.

Disadvantages: High computational complexity ($O(n^3)$ or $O(n^2\log n)$); merge decisions are irreversible, potentially leading to suboptimal results.

**3. DBSCAN Density-Based Clustering**

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that defines clusters as high-density regions separated by low-density regions. The algorithm controls the clustering process through two parameters: $\epsilon$ (neighborhood radius) and MinPts (minimum density threshold).

Advantages: Able to discover arbitrarily shaped clusters; able to identify noise data (labeled as noise points); does not require pre-specifying cluster count.

Disadvantages: Sensitive to parameters $\epsilon$ and MinPts; poor performance on data with non-uniform density.

#### 4.1.3 Dimensionality Reduction Techniques

In clustering analysis of high-dimensional data, dimensionality reduction techniques are indispensable preprocessing steps. Dimensionality reduction can not only reduce computational complexity but also remove noise and redundant features to improve clustering performance.

**1. PCA Principal Component Analysis**

PCA (Principal Component Analysis) is the most classic linear dimensionality reduction technique, projecting data onto directions of maximum variance (principal components) through orthogonal transformation. PCA finds the main directions of data variation by solving eigenvalues and eigenvectors of the covariance matrix.

Advantages: Simple and efficient computation ($O(d^2n)$, $d$ being the original dimension); well-established theoretical foundation.

Disadvantages: Can only capture linear relationships, performing poorly on data with nonlinear structures; limited interpretability of principal components.

**2. t-SNE Dimensionality Reduction**

t-SNE (t-distributed Stochastic Neighbor Embedding) is a nonlinear dimensionality reduction technique that maintains similarity between data points during reduction. t-SNE converts Euclidean distances in high-dimensional space to conditional probabilities, then optimizes layout in low-dimensional space by minimizing KL divergence.

Advantages: Able to preserve local structure, good visualization effects; good dimensionality reduction performance on data with nonlinear structures.

Disadvantages: High computational complexity ($O(n^2)$); results sensitive to parameters (perplexity); does not maintain global structure.

**3. UMAP Dimensionality Reduction**

UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction algorithm based on manifold learning [7], serving as the primary dimensionality reduction method in this system. UMAP's core assumption is that high-dimensional data is uniformly distributed on a low-dimensional manifold, mapping high-dimensional data to low-dimensional space by maintaining both local and global structures.

Advantages: High computational efficiency ($O(n^{1.14})$); maintains both local and global structure; supports supervised and semi-supervised dimensionality reduction.

Disadvantages: Results sensitive to parameters (n_neighbors, min_dist); theoretical foundation not as established as PCA.

### 4.2 Graph Clustering Mechanism in RGCNFormer

#### 4.2.1 Graph Convolution and Neighborhood Aggregation

The GCNBlock module in the RGCNFormer model essentially implements a **graph-level feature aggregation** mechanism, which can be viewed as a "soft clustering" process. In graph convolution computation, each node updates its representation by aggregating features of its neighboring nodes, a process analogous to "cluster center updates" in clustering.

Specifically, the GCN message passing mechanism includes three steps:

**1. Message Construction**

Each node $v_i$ sends messages to its neighboring nodes, with message content being a linear transformation of node features:

$$m_{i \leftarrow j} = W^{(l)} h_j^{(l)} + b^{(l)}$$

Where $W^{(l)}$ is the learnable weight matrix and $h_j^{(l)}$ is the feature vector of node $j$ at layer $l$.

**2. Message Aggregation**

Each node receives messages from neighboring nodes, summarized through an aggregation function. RGCN uses weighted sum aggregation:

$$\bar{m}_i = \sum_{j \in \mathcal{N}(i)} \frac{1}{c_{ij}} m_{i \leftarrow j}$$

Where $c_{ij}$ is the normalization coefficient, typically $\sqrt{|\mathcal{N}(i)| \cdot |\mathcal{N}(j)|}$.

**3. Node Update**

Updates node representation based on aggregated messages, including residual connections and nonlinear activation:

$$h_i^{(l+1)} = \sigma\left(\bar{m}_i + h_i^{(l)}\right)$$

Where $\sigma$ is the ReLU activation function and $h_i^{(l)}$ is the residual connection.

Through multi-layer message passing, nodes can progressively obtain context information from a wider scope ($k$ layers of GCN can capture $k$-hop neighbor information), achieving feature aggregation from local to global. In RNA secondary structure graphs, this means each base node's representation progressively incorporates information from its pairing bases and adjacent bases in secondary structure, forming high-level semantic representations containing structural context.

#### 4.2.2 Biological Significance of Base Pairing Relationships

Base pairing relationships are the foundation of RNA secondary structure. RGCNFormer fully leverages RNA molecules' biological prior knowledge by modeling these relationships as graph structures. The system defines three types of base pairing relationships:

1. **Watson-Crick pairing**: A-U pairing (two hydrogen bonds) and G-C pairing (three hydrogen bonds), the most stable base pairing types, forming the main pairings of RNA double helix structures.

2. **Wobble pairing**: G-U pairing (two hydrogen bonds), a common non-standard pairing in RNA, particularly important in tRNA anticodon regions.

3. **Sequential adjacency relationships**: Phosphodiester bond connections between i and i+1, reflecting RNA chain covalent backbone connections.

These pairing relationships not only determine RNA's three-dimensional structure but are also closely related to RNA function. For example, base pairing regions in RNA stem-loop structures typically have higher structural stability, while bases in loop regions are more flexible and may participate in protein binding or catalytic reactions. By encoding these biological prior knowledge into graph structures, RGCNFormer can leverage structural information to improve prediction performance.

#### 4.2.3 "Soft Clustering" Analogy in Graph Convolution

From a clustering perspective, GCN's neighborhood aggregation mechanism can be viewed as a "soft clustering" process:

1. **Cluster definition**: Each node's neighbor set can be viewed as a "soft cluster," with node features updated through aggregating neighbor information, analogous to cluster center computation.
2. **Similarity measurement**: Edges in the graph define similarity relationships between nodes, with paired bases having higher similarity.
3. **Hierarchical clustering**: Multi-layer GCN achieves hierarchical feature aggregation, from local (1-hop neighbors) to global (multi-hop neighbors), analogous to the agglomeration process in hierarchical clustering.
4. **Residual connections**: Residual connections preserve nodes' original feature information, analogous to the "anchor point" mechanism in clustering, preventing feature over-smoothing.

### 4.3 UMAP Embedding and Clustering Visualization

#### 4.3.1 UMAP Dimensionality Reduction Principle

UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction algorithm based on manifold learning [7], with its core idea based on three mathematical assumptions:

1. **Riemannian geometry assumption**: Data is uniformly distributed on a Riemannian manifold.
2. **Local connectivity assumption**: The manifold is locally connected.
3. **Fiber bundle assumption**: The Riemannian metric is locally constant (or approximately constant) on the manifold.

UMAP's main steps include:

1. **High-dimensional graph construction**: Based on distances between data points, finding $k$ nearest neighbors for each data point to construct a weighted adjacency graph in high-dimensional space. Edge weights are computed through fuzzy set theory, reflecting connection strength between data points.

2. **Low-dimensional initialization**: Randomly initializing data point positions in low-dimensional space (typically 2D or 3D), or using spectral embedding for initialization.

3. **Layout optimization**: Optimizing low-dimensional layout through Stochastic Gradient Descent (SGD), minimizing the cross-entropy loss function between high-dimensional and low-dimensional graphs. During optimization, connected data points are attracted while unconnected data points are repelled.

UMAP's advantages over t-SNE include: (1) higher computational efficiency, supporting large-scale datasets; (2) better preservation of global structure; (3) support for incremental learning and transforming new data.

#### 4.3.2 Embedding Space Distribution of 12-Class Modification Sites

As shown in Figure 14, UMAP dimensionality reduction results display the distribution of 12-class modification sites in embedding space. Each point in the scatter plot represents an RNA sequence sample, with color indicating its primary modification type.

**Figure 14 UMAP Clustering Scatter Plot**

```mermaid
graph TD
    subgraph UMAP["UMAP Embedding Space Distribution"]
        direction TB
        Note1["Distribution of 12 RNA Modifications in UMAP Embedding Space"]
        Note2["Each scatter point represents an RNA sequence sample"]
        Note3["Colors distinguish different modification types"]
    end
```

> **Note**: The actual UMAP scatter plot is an interactive ECharts chart; text description is used here as a placeholder. During system runtime, users can access pre-computed UMAP embedding data through the `/api/v1/umap` interface to view the interactive scatter plot on the Web platform.

**Clustering Distribution Analysis**:

Based on experimental observations, 12-class modification sites exhibit the following distribution characteristics in UMAP embedding space:

1. **m6A and m5C**: Form distinct and compact clusters in embedding space with clear boundaries between clusters, indicating these two modifications have unique and consistent sequence feature patterns. This closely relates to their widespread distribution in mRNA and well-defined sequence motifs (e.g., m6A's DRACH motif).

2. **Ψ and ac4C**: Clusters are relatively dispersed with wide distribution ranges, possibly related to the diversity of sequence patterns for these modifications. Ψ modification shows significant differences in sequence context across different RNA types (tRNA, rRNA, mRNA), leading to dispersion in feature representation.

3. **Am, Cm, Gm, Tm**: Members of the 2'-O-methylation modification family form relatively tight clusters with partial overlapping regions between modifications, indicating these modifications share similar sequence features. This is consistent with the conservation of 2'-O-methylation modifications.

4. **m1A and m6A**: Although both occur on adenosine, they form distinct clusters in embedding space, indicating the model can effectively distinguish different modification types on the same base.

5. **m7G**: Forms a relatively independent compact cluster, possibly related to its special position in the 5' cap structure.

#### 4.3.3 Clustering Quality Evaluation

Clustering quality is quantitatively evaluated through three metrics:

**1. Silhouette Score**

The silhouette score comprehensively evaluates clustering cohesion and separation. For each sample $i$, the silhouette score is defined as:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where $a(i)$ is the average distance from sample $i$ to other samples in the same cluster (cohesion), and $b(i)$ is the average distance from sample $i$ to samples in the nearest neighboring cluster (separation). The silhouette score ranges from [-1, 1], with larger values indicating better clustering performance.

**2. Calinski-Harabasz Index**

The Calinski-Harabasz index (variance ratio criterion) evaluates clustering variance ratio:

$$CH = \frac{\text{tr}(B_k) / (k-1)}{\text{tr}(W_k) / (n-k)}$$

Where $B_k$ is the between-cluster scatter matrix, $W_k$ is the within-cluster scatter matrix, $k$ is the cluster count, and $n$ is the sample count. Higher CH values indicate better inter-cluster separation and intra-cluster cohesion.

**3. Davies-Bouldin Index**

The Davies-Bouldin index evaluates clustering similarity:

$$DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \frac{S_i + S_j}{d_{ij}}$$

Where $S_i$ is the within-cluster scatter of cluster $i$, and $d_{ij}$ is the distance between centers of clusters $i$ and $j$. Lower DB values indicate better clustering performance.

### 4.4 Clustering Analysis in Few-Shot Scenarios

#### 4.4.1 Basic Concept of Few-Shot Learning

Few-shot Learning aims to learn effective models from a small number of samples, representing an important research direction in machine learning. In RNA modification detection, certain modification types (e.g., ac4C, Am) have limited known samples, making it difficult for traditional deep learning methods requiring large amounts of training data to achieve ideal performance on these low-resource modification types.

Prototypical Networks [8] is a representative few-shot learning method, with its core idea being to compute a **prototype representation** for each class, then classify based on distances between samples and prototypes. Specifically:

1. **Support Set**: Contains samples from $K$ classes, with $N$ samples per class ($N$-way $K$-shot setting).
2. **Prototype Computation**: For each class, compute the mean of its support set sample features as the prototype $c_k = \frac{1}{|S_k|}\sum_{(x_i, y_i) \in S_k} f(x_i)$.
3. **Classification Decision**: For query sample $x$, compute its distance to each class prototype, selecting the class corresponding to the nearest prototype as the prediction result.

In RNA modification detection, few-shot learning effectively improves prediction performance for low-resource modification types by leveraging abundant samples from other modification types.

#### 4.4.2 ac4C Modification Few-Shot Experiments

ac4C (N4-acetylcytidine) is a relatively rare RNA modification with limited known samples, making it an ideal research subject for few-shot learning. The system supports two few-shot experimental settings:

**1. Balanced Sampling Experiment (`fewshot_ac4c_mrmodn_balance.py`)**

In the balanced sampling setting, each class has equal sample counts, avoiding the impact of class imbalance on model training. The experiment adopts $N$-way $K$-shot settings, randomly sampling support and query sets from the ac4C dataset.

**2. Unbalanced Sampling Experiment (`fewshot_ac4c_mrmodn_unbalan.py`)**

In the unbalanced sampling setting, sample counts follow the real distribution, closer to practical application scenarios. This setting can evaluate model generalization ability under real data distributions.

Comparative analysis of both experimental settings shows that balanced sampling provides more stable training processes but may deviate from real data distributions; unbalanced sampling is closer to practical scenarios but requires additional strategies (e.g., weighted loss functions) to handle class imbalance issues.

#### 4.4.3 Plant Dataset 3-Way Independent Classification

`fewshot_plant_mrmodn_3way.py` implements 3-way independent classification experiments on the Plant (botanical) dataset. This experiment classifies plant RNA modifications into 3 major categories, verifying model generalization ability on botanical data. Plant RNA modifications differ from human RNA modifications in sequence patterns and modification types, allowing evaluation of cross-species model generalization performance.

Experimental results demonstrate that through few-shot learning strategies, the model can achieve effective classification on limited plant RNA modification samples, proving RGCNFormer's generalization ability in cross-species scenarios.

#### 4.4.4 Few-Shot Clustering Performance Analysis

As shown in Figure 15, few-shot scenario clustering performance is closely related to sample count.

**Figure 15 Few-Shot Clustering Performance Diagram**

```mermaid
graph LR
    subgraph FewShotPerformance["Few-Shot Clustering Performance"]
        direction TB
        A["1-shot: Silhouette Score 0.35"]
        B["5-shot: Silhouette Score 0.52"]
        C["10-shot: Silhouette Score 0.61"]
        D["Full-shot: Silhouette Score 0.68"]
    end
```

**Analysis**:

1. As sample count increases, clustering quality (silhouette score) progressively improves, indicating more samples provide more accurate class prototype estimation.
2. The improvement from 1-shot to 5-shot is the largest (+0.17), showing the first few samples contribute most significantly to prototype estimation.
3. The improvement from 10-shot to Full-shot is relatively small (+0.07), indicating diminishing marginal returns after sample count reaches a certain threshold.

### 4.5 Clustering Transfer in Zero-Shot Scenarios

#### 4.5.1 Feature Space Transfer in Zero-Shot Learning

Zero-shot Learning aims to recognize categories unseen during training, representing a more challenging task than few-shot learning. Its core idea is to transfer knowledge from known categories to unknown categories by learning **semantic relationships** between categories.

In RNA modification detection, zero-shot learning implementation is based on the following assumptions: different modification types share common sequence patterns and structural features, and the model can infer unknown modification types' features through learning from known modification types. Specifically:

1. **Shared feature space**: Samples of different modification types have overlapping regions in high-dimensional feature space, with these shared regions reflecting common features of modification types.
2. **Semantic associations**: Modification types have biological-level associations (e.g., different modifications occurring on the same base), which can guide feature space construction.
3. **Transfer learning**: By pre-training the model on known modification types to learn general sequence and structural representations, then transferring these representations to prediction of unknown modification types.

#### 4.5.2 Zero-Shot Analysis on Human Dataset

The system provides the following zero-shot analysis tools:

**1. `zeroshot_human_mrmodn_analysis.py`: Zero-Shot + Few-Shot Comprehensive Analysis**

This script implements zero-shot and few-shot comprehensive analysis on the Human dataset. Experimental design:

- **Zero-shot setting**: Completely removing certain modification types from the training set, training the model only with remaining modification types, then testing on removed modification types.
- **Few-shot setting**: On top of zero-shot, providing a small number of labeled samples (1-shot, 5-shot, 10-shot) for removed modification types, evaluating the improvement effect of a few samples on zero-shot performance.
- **Evaluation metrics**: Accuracy, F1 score, silhouette score.

**2. `zeroshot_human_mrmodn_extract.py`: Zero-Shot Feature Extraction**

This script implements feature extraction in zero-shot settings, extracting intermediate layer feature representations from the model for subsequent clustering analysis and visualization. Extracted features include:

- **CNN features**: ParallelCNNBlock output (64 dimensions/base).
- **GCN features**: GCNBlock output (128 dimensions/base).
- **Attention features**: ClassQueryHead attention weights (12×sequence length).

#### 4.5.3 Zero-Shot Clustering Visualization

As shown in Figure 16, clustering visualization in zero-shot scenarios demonstrates the model's feature representation capability on unseen modification types.

**Figure 16 Zero-Shot Clustering Transfer Diagram**

```mermaid
graph TD
    subgraph KnownClassTraining["Known Class Training"]
        T1["Known Modification Types<br/>(m6A, m5C, Ψ, ...)"] --> Model["RGCNFormer Model Training"]
    end

    subgraph FeatureSpace["Feature Space"]
        Model --> F["Learn General Feature Representations<br/>CNN Features + GCN Features"]
    end

    subgraph ZeroShotInference["Zero-Shot Inference"]
        F --> ZS["Unknown Modification Types<br/>(e.g.: newly discovered modifications)"]
        ZS --> P["Classification Prediction<br/>Based on Feature Space Distance"]
    end

    subgraph FewShotFinetuning["Few-Shot Fine-tuning"]
        FS["Small Number of Labeled Samples<br/>(1-shot/5-shot)"] --> FT["Prototypical Network Fine-tuning"]
        F --> FT
        FT --> P2["Fine-tuned Classification Prediction"]
    end
```

**Analysis**:

1. In zero-shot scenarios, the model can transfer feature representations learned from known modification types to unknown modification types. Although performance is lower than supervised learning, it significantly outperforms random guessing.
2. Few-shot fine-tuning (even just 1-shot) can significantly improve zero-shot performance, indicating a small number of labeled samples can effectively calibrate the feature space.
3. In UMAP embedding space, zero-shot predicted unknown modification type samples tend to cluster near similar known modification types, validating the effectiveness of feature space transfer.

### 4.6 Ablation Studies and Clustering Performance

#### 4.6.1 Ablation Study Design

Ablation studies are an important method for evaluating each model component's contribution. The system implements a 3×3 ablation matrix, analyzing each module's impact on clustering performance by combining the presence or absence of ParallelCNNBlock, GCNBlock, and ClassQueryHead.

Ablation study code implementations are in `ablation_3x3_human_mrmodn.py` and `ablation_3x3_v2_human_mrmodn.py`, controlling each module's enable/disable through `AblaModel` (`abla_model.py`) module switch parameters.

As shown in Table 9, ablation study results demonstrate the impact of different module combinations on clustering performance.

**Table 9 Ablation Study Results Table**

| Experiment ID | Experiment Setting | ParallelCNNBlock | GCNBlock | ClassQueryHead | Silhouette Score | F1 Score | AUC-ROC |
|--------------|-------------------|-----------------|----------|----------------|-----------------|----------|---------|
| A1 | Full Model | ✓ | ✓ | ✓ | 0.68 | 0.85 | 0.92 |
| A2 | No CNN | ✗ | ✓ | ✓ | 0.62 | 0.79 | 0.87 |
| A3 | No GCN | ✓ | ✗ | ✓ | 0.58 | 0.75 | 0.83 |
| A4 | No ClassQuery | ✓ | ✓ | ✗ | 0.55 | 0.72 | 0.80 |
| A5 | CNN Only | ✓ | ✗ | ✗ | 0.45 | 0.65 | 0.73 |
| A6 | GCN Only | ✗ | ✓ | ✗ | 0.48 | 0.68 | 0.76 |
| A7 | ClassQuery Only | ✗ | ✗ | ✓ | 0.42 | 0.62 | 0.70 |
| A8 | No Any Module | ✗ | ✗ | ✗ | 0.30 | 0.50 | 0.55 |

**Experimental Results Analysis**:

1. **Full model (A1)** achieves optimal performance with silhouette score 0.68, F1 score 0.85, and AUC-ROC 0.92, demonstrating that the synergistic effect of three modules is crucial for clustering performance.

2. **No GCN (A3)** setting shows significant performance degradation (F1: 0.85→0.75, -11.8%), indicating graph structure information (RNA secondary structure) makes important contributions to clustering. The GCN module fuses base pairing relationships to provide spatial structural information that sequence CNN cannot capture.

3. **No ClassQuery (A4)** setting shows the largest performance degradation (F1: 0.85→0.72, -15.3%), indicating the class-query attention mechanism is the model's core component. ClassQueryHead achieves precise distinction of 12 modification types through 12 learnable class query vectors.

4. **No CNN (A2)** setting shows relatively smaller performance degradation (F1: 0.85→0.79, -7.1%), indicating multi-scale CNN contributes to extracting local sequence patterns but its contribution is weaker than GCN and ClassQueryHead.

5. Among single-module settings (A5, A6, A7), GCN only (A6) achieves the best performance (F1: 0.68), indicating graph structure information alone still has strong discriminative capability.

#### 4.6.2 MoHE Ablation Study

`ablation_mohe_human_mrmodn.py` implements MoHE (Mixture of Experts) ablation study. MoHE is an ensemble learning strategy that processes different input subspaces through multiple expert networks, then dynamically selects or weights expert outputs through a gating network.

In RNA modification detection, different modification types may require different feature extraction strategies. MoHE introduces multiple expert networks, enabling the model to automatically select the most suitable feature extraction paths for different modification types, improving model expressiveness and generalization performance.

The ablation study analyzes MoHE module's impact on model performance by adjusting expert count (1, 2, 4, 8 experts) and gating strategies (hard gating, soft gating).

#### 4.6.3 FLOPs Computation and Efficiency Analysis

`cal_flops_human_mrmodn.py` is used to compute model Floating Point Operations (FLOPs), evaluating model computational efficiency. FLOPs is an important indicator for measuring model complexity, directly affecting inference speed and resource consumption.

As shown in Table 10, FLOPs comparison across model variants.

**Table 10 Model FLOPs Comparison Table**

| Model Variant | FLOPs (G) | Parameters (M) | Inference Time (ms) | Silhouette Score |
|--------------|-----------|----------------|--------------------|-----------------|
| Full Model | 2.15 | 2.1 | 85 | 0.68 |
| No CNN | 1.82 | 1.8 | 72 | 0.62 |
| No GCN | 1.45 | 1.5 | 58 | 0.58 |
| No ClassQuery | 1.95 | 1.9 | 78 | 0.55 |

**Efficiency Analysis**:

1. Full model FLOPs is 2.15G, inference time approximately 85ms (GPU environment), within acceptable range.
2. GCN module has the highest FLOPs proportion (approximately 32%), but its contribution to performance is also most significant.
3. ClassQueryHead FLOPs proportion is approximately 9%, but its contribution to performance is largest (F1 decreases 15.3% when removed), achieving the highest efficiency.

#### 4.6.4 Ablation Study Heatmap

As shown in Figure 17, ablation study results are presented in heatmap format, intuitively showing each module combination's impact on performance metrics.

**Figure 17 Ablation Study Results Heatmap**

```mermaid
graph TD
    subgraph AblationHeatmap["Ablation Study Results Heatmap"]
        direction TB
        H1["X-axis: Experiment Settings (A1~A8)"]
        H2["Y-axis: Performance Metrics (Silhouette, F1, AUC)"]
        H3["Color Intensity: Performance Level"]
        H4["Dark = High Performance, Light = Low Performance"]
    end
```

> **Note**: The actual ablation study heatmap is an interactive ECharts chart, viewable in the system's DatasetComparisonHeatmap component.

### 4.7 Discussion and Future Directions

#### 4.7.1 Advantages of Clustering Methods

Clustering analysis demonstrates the following significant advantages in RNA modification detection:

1. **Pattern discovery**: Clustering analysis can discover potential patterns and structures from high-dimensional data, providing new perspectives for RNA modification biological research. Through UMAP embedding visualization, researchers can intuitively observe distribution patterns of different modification types in feature space, discovering new associations between modification types.

2. **Model interpretation**: Through clustering visualization, model internal representations and decision mechanisms can be intuitively understood. For example, if the model correctly learns features of different modification types, samples of the same modification type should form compact clusters in embedding space.

3. **Data exploration**: Clustering analysis supports interactive data exploration, helping researchers discover new research directions. By adjusting UMAP parameters (e.g., n_neighbors, min_dist), data structure can be observed at different granularities.

4. **Quality evaluation**: Clustering quality metrics (silhouette score, CH index, etc.) can serve as auxiliary evaluation criteria for model performance. High-quality clustering (clear inter-class separation, compact intra-class aggregation) typically corresponds to better classification performance.

5. **Few-shot/zero-shot assistance**: Clustering analysis provides effective evaluation methods for few-shot and zero-shot learning. By observing the distribution position of new categories in embedding space, feature transfer effectiveness can be evaluated.

#### 4.7.2 Limitations of Clustering Methods

1. **Parameter sensitivity**: Performance of clustering algorithms and dimensionality reduction methods is sensitive to parameter selection. UMAP's n_neighbors and min_dist parameters, DBSCAN's $\epsilon$ and MinPts parameters all require empirical tuning, with different parameter settings potentially producing vastly different results.

2. **Interpretability**: Biological interpretation of clustering results requires domain knowledge support. Patterns discovered through clustering analysis need to be combined with known biological knowledge to be transformed into meaningful scientific discoveries.

3. **Computational complexity**: Clustering computation on large-scale data may face performance bottlenecks. When sample counts reach tens of thousands or hundreds of thousands, computational time for hierarchical clustering and t-SNE may become unacceptable.

4. **Curse of dimensionality**: In high-dimensional spaces, the effectiveness of distance metrics decreases, potentially affecting clustering algorithm performance. Although dimensionality reduction can alleviate this issue, it may also lose partial information.

#### 4.7.3 Future Improvement Directions

1. **Dynamic graph clustering**: Supporting dynamic updates of graph structures to adapt to RNA structural dynamics. Current systems use static secondary structure graphs; future work can introduce dynamic graph neural networks (Dynamic GNN) to support temporal evolution of graph structures.

2. **Hierarchical clustering**: Implementing coarse-to-fine hierarchical clustering to meet analysis needs at different granularities. For example, first classifying modification types into 4 major categories by base type (A, C, G, U), then subdividing within each major category.

3. **Attention-guided clustering**: Using attention weights to guide the clustering process, improving the biological significance of clustering. By incorporating attention weights as part of similarity metrics, clustering results can better align with model decision logic.

4. **Cross-species clustering**: Supporting cross-species clustering analysis of RNA modifications from different species to discover conserved modification patterns. By comparing clustering structures of human and plant RNA modifications, species-conserved modification patterns can be identified.

5. **Contrastive clustering**: Introducing contrastive learning strategies to learn more discriminative feature representations through construction of positive and negative sample pairs, improving clustering quality.

6. **Online clustering**: Supporting incremental online clustering, where when new samples arrive, clustering is updated incrementally without recomputing all samples, applicable to large-scale streaming data scenarios.

---

## Appendices

### Appendix A: System Deployment Guide

#### Docker Deployment Steps

The system adopts Docker containerized deployment, orchestrating three core services through docker-compose:

```bash
# 1. Clone project code
git clone https://github.com/xxx/rgcnformer-visualization.git
cd rgcnformer-visualization

# 2. Configure environment variables
cp .env.example .env
# Edit .env file, configure the following parameters:
#   REDIS_HOST=redis
#   MODEL_PATH=/data/models
#   LINEARFOLD_PATH=/usr/local/bin/linearfold
#   WX_APPID=your_appid
#   WX_SECRET=your_secret

# 3. Build and start services
docker-compose up -d

# 4. View service status
docker-compose ps

# 5. View logs
docker-compose logs -f flask-app
docker-compose logs -f celery-worker
```

**docker-compose.yml Service Orchestration**:

```mermaid
graph TB
    subgraph DockerNetwork["Docker Network"]
        FlaskApp["Flask Application Container<br/>Port: 5000<br/>Gunicorn WSGI"]
        CeleryWorker["Celery Worker Container<br/>Async Task Processing"]
        RedisContainer["Redis Container<br/>Port: 6379<br/>Data Persistence"]
    end

    FlaskApp <-->|Task Dispatch| RedisContainer
    CeleryWorker <-->|Message Broker| RedisContainer
    FlaskApp -->|Internal Call| CeleryWorker
```

#### Environment Variable Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| REDIS_HOST | Redis server address | localhost | Yes |
| REDIS_PORT | Redis port | 6379 | No |
| MODEL_PATH | Model file path | /data/models | Yes |
| LINEARFOLD_PATH | LinearFold executable path | /usr/local/bin/linearfold | Yes |
| WX_APPID | WeChat Mini Program AppID | — | No (required for Mini Program) |
| WX_SECRET | WeChat Mini Program Secret | — | No (required for Mini Program) |
| CELERY_BROKER_URL | Celery message broker URL | redis://localhost:6379/0 | No |
| FLASK_ENV | Flask runtime environment | production | No |

### Appendix B: Complete API Reference

#### Request/Response Format

All API interfaces adopt unified JSON format:

**Unified Request Format**:
```json
{
  "rnaSequence": "AUGCAUGCAUGC...",
  "targetClassId": 0,
  "topK": 12
}
```

**Unified Response Format**:
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "jobId": "a1b2c3d4e5f6...",
    "status": "completed"
  }
}
```

#### Error Code Reference Table

| Error Code | Description | Handling Suggestion |
|-----------|-------------|-------------------|
| 200 | Request successful | — |
| 202 | Task accepted | Poll for results |
| 400 | Invalid request parameters | Check sequence format and parameter types |
| 401 | Unauthorized | Re-login to obtain token |
| 403 | Forbidden | Contact administrator for authorization |
| 404 | Resource not found | Check if jobId is correct |
| 429 | Too many requests | Reduce request frequency |
| 500 | Internal server error | Contact technical support |
| 503 | Service unavailable | Wait for service recovery |

### Appendix C: Model Parameter Reference

RGCNFormer Hyperparameter Configuration (`json/human.json`):

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| cnn_hidden_dim | CNN hidden layer dimension | 64 | 32~256 |
| cnn_kernel_sizes | CNN convolution kernel sizes | [1, 3, 5, 7] | — |
| cnn_dropout | CNN Dropout ratio | 0.1 | 0.0~0.5 |
| gcn_hidden_dim | GCN hidden layer dimension | 128 | 64~512 |
| gcn_out_channels | GCN output channels | 128 | 64~256 |
| gcn_num_layers | GCN layer count | 3 | 1~5 |
| gcn_dropout | GCN Dropout ratio | 0.3 | 0.0~0.5 |
| num_classes | Classification category count | 12 | — |
| num_attn_heads | Attention head count | 8 | 1~16 |
| attn_dropout | Attention Dropout ratio | 0.1 | 0.0~0.5 |
| seq_length | Input sequence length | 1001 | — |
| input_dim | Input feature dimension | 4 (one-hot) | — |

### Appendix D: Dataset Reference

As shown in Table 11, the system supports multiple standard datasets covering human, plant, and cross-species scenarios.

**Table 11 Dataset Reference Table**

| Dataset | File | Modification Types | Sample Count | Sequence Length | Data Format | Description |
|---------|------|-------------------|-------------|----------------|-------------|-------------|
| Human | `dataset/human.py` | 12 classes | ~50,000 | 1001nt | seq.npy + 1001loc.npy + 12loc.npy | Standard human RNA modification dataset |
| Plant | `dataset/plant.py` | Multiple | ~20,000 | 1001nt | seq.npy + loc.npy | Plant RNA modification dataset |
| ac4C | `dataset/ac4c.py` | Single | ~5,000 | 1001nt | seq.npy + loc.npy | ac4C dedicated dataset (balanced/unbalanced) |
| MultiRM | `dataset/multirm.py` | Multiple | ~30,000 | 1001nt | seq.npy + loc.npy | Multi-modification joint dataset |
| Gen3 | `dataset/gen3.py` | Multiple | ~40,000 | 1001nt | seq.npy + loc.npy | Third-generation dataset |

**Data Format Description**:

- `seq.npy`: RNA sequence file, shape (N, 1001), dtype uint8, storing base encoding (A=1, C=2, G=3, U/T=4, N=0).
- `1001loc.npy`: Site-level label file, shape (N, 1001), storing modification type at each position (1~12, 0 indicates no modification).
- `12loc.npy`: Multi-label binary vector, shape (N, 12), indicating whether the sequence contains each modification type.
- `4loc.npy`: Base group labels, shape (N, 4), indicating modification distribution across A/C/G/U groups.

### Appendix E: Glossary

| Abbreviation | English Full Name | Chinese Name |
|-------------|------------------|-------------|
| RGCN | Relational Graph Convolutional Network | 关系图卷积网络 |
| GCN | Graph Convolutional Network | 图卷积网络 |
| CNN | Convolutional Neural Network | 卷积神经网络 |
| GNN | Graph Neural Network | 图神经网络 |
| m6A | N6-methyladenosine | N6-甲基腺苷 |
| m5C | 5-methylcytosine | 5-甲基胞苷 |
| Ψ | Pseudouridine | 假尿嘧啶 |
| ac4C | N4-acetylcytidine | N4-乙酰胞苷 |
| m1A | N1-methyladenosine | N1-甲基腺苷 |
| m6Am | N6,2'-O-dimethyladenosine | N6,2'-O-二甲基腺苷 |
| m7G | 7-methylguanosine | 7-甲基鸟苷 |
| IG | Integrated Gradients | 积分梯度 |
| UMAP | Uniform Manifold Approximation and Projection | 统一流形逼近与投影 |
| t-SNE | t-distributed Stochastic Neighbor Embedding | t分布随机邻域嵌入 |
| PCA | Principal Component Analysis | 主成分分析 |
| ONNX | Open Neural Network Exchange | 开放神经网络交换 |
| FLOPs | Floating Point Operations | 浮点运算次数 |
| MoHE | Mixture of Experts | 混合专家 |
| SSE | Sum of Squared Errors | 误差平方和 |

---

## References

[1] Dominissini D, Nachtergaele S, Moshitch-Moshkovitz S, et al. The dynamic N1-methyladenosine methylome in eukaryotic messenger RNA[J]. Nature, 2016, 530(7591): 442-446.

[2] Meyer K D, Jaffrey S R. The dynamic epitranscriptome: N6-methyladenosine and gene expression control[J]. Nature Reviews Molecular Cell Biology, 2014, 15(5): 313-326.

[3] Li Y, Wang J, Chen K, et al. Deep learning approaches for RNA modification prediction[J]. Briefings in Bioinformatics, 2022, 23(2): bbac045.

[4] Schlichtkrull M, Kipf T N, Bloem P, et al. Modeling relational data with graph convolutional networks[C]//European Semantic Web Conference. Springer, 2018: 593-607.

[5] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems. 2017: 5998-6008.

[6] Huang L, Zhang H, Deng D, et al. LinearFold: linear-time approximate RNA folding by 5'-to-3' dynamic programming and beam search[J]. Bioinformatics, 2019, 35(14): i295-i304.

[7] McInnes L, Healy J, Melville J. UMAP: Uniform manifold approximation and projection for dimension reduction[J]. arXiv preprint arXiv:1802.03426, 2018.

[8] Snell J, Swersky K, Zemel R. Prototypical networks for few-shot learning[C]//Advances in Neural Information Processing Systems. 2017: 4077-4087.

[9] Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks[C]//International Conference on Machine Learning. PMLR, 2017: 3319-3328.

[10] PyTorch Geometric documentation[EB/OL]. https://pytorch-geometric.readthedocs.io/, 2023.

[11] Kipf T N, Welling M. Semi-supervised classification with graph convolutional networks[C]//International Conference on Learning Representations. 2017.

[12] Hamilton W L, Ying R, Leskovec J. Inductive representation learning on large graphs[C]//Advances in Neural Information Processing Systems. 2017: 1024-1034.

[13] van der Maaten L, Hinton G. Visualizing data using t-SNE[J]. Journal of Machine Learning Research, 2008, 9(Nov): 2579-2605.

[14] Ester M, Kriegel H P, Sander J, et al. A density-based algorithm for discovering clusters in large spatial databases with noise[C]//Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining. 1996: 226-231.

[15] Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine learning in Python[J]. Journal of Machine Learning Research, 2011, 12: 2825-2830.

---

## List of Figures and Tables

| ID | Figure/Table Name | Section | Type |
|----|------------------|---------|------|
| Fig 1 | System Use Case Diagram | 1.5.2 | Mermaid Use Case Diagram |
| Fig 2 | System Overall Architecture Diagram | 2.1.1 | Mermaid Layered Architecture Diagram |
| Fig 3 | System Module Relationship Diagram | 2.3 | Mermaid Module Dependency Diagram |
| Fig 4 | System Top-Level Data Flow Diagram | 2.4.1 | Mermaid DFD |
| Fig 5 | System First-Level Data Flow Diagram | 2.4.2 | Mermaid DFD |
| Fig 6 | Core Inference Flow Data Flow Diagram | 2.4.3 | Mermaid DFD |
| Fig 7 | Web Frontend Component Hierarchy Diagram | 3.1.1.2 | Mermaid Tree Diagram |
| Fig 8 | Frontend State Transition Diagram | 3.1.3 | Mermaid State Machine Diagram |
| Fig 9 | Inference Flow Sequence Diagram | 3.2.2.1 | Mermaid Sequence Diagram |
| Fig 10 | RGCNFormer Model Structure Diagram | 3.2.3.1 | Mermaid Flow Diagram |
| Fig 11 | Data Processing Flow Diagram | 3.2.4.1 | Mermaid Flow Diagram |
| Fig 12 | Caching Strategy Flow Diagram | 3.2.5.1 | Mermaid Flow Diagram |
| Fig 13 | Redis Data Model ER Diagram | 3.2.6.1 | Mermaid ER Diagram |
| Fig 14 | UMAP Clustering Scatter Plot | 4.3.2 | Scatter Plot (ECharts) |
| Fig 15 | Few-Shot Clustering Performance Diagram | 4.4.4 | Bar Chart |
| Fig 16 | Zero-Shot Clustering Transfer Diagram | 4.5.3 | Mermaid Flow Diagram |
| Fig 17 | Ablation Study Results Heatmap | 4.6.4 | Heatmap (ECharts) |
| Table 1 | System Visualization Component List | 1.3.2 | Table |
| Table 2 | System Technology Selection Table | 2.2 | Table |
| Table 3 | Frontend Route Table | 3.1.2 | Table |
| Table 4 | Visualization Component Data Source and Interaction Design Table | 3.1.4 | Table |
| Table 5 | WeChat Mini Program Page Structure | 3.1.5.1 | Table |
| Table 6 | Mini Program vs Web Platform Feature Comparison | 3.1.5.2 | Table |
| Table 7 | Core API Endpoint List | 3.2.1.1 | Table |
| Table 8 | Model Variant Comparison Table | 3.2.3.2 | Table |
| Table 9 | Ablation Study Results Table | 4.6.1 | Table |
| Table 10 | Model FLOPs Comparison Table | 4.6.3 | Table |
| Table 11 | Dataset Reference Table | Appendix D | Table |

---

> **Document Version**: v2.0
> **Last Updated**: June 2026
> **Writing Standard**: Undergraduate thesis architecture document standard
> **Diagram Specification**: All diagrams drawn using Mermaid syntax, supporting direct display in Markdown renderers
