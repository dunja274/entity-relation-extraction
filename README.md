# Document-Level Entity Relation Extraction

This GitHub repository contains the codebase and documentation for a comprehensive study on document-level entity relation extraction, conducted as part of a thesis. The research focuses on leveraging state-of-the-art generative models, with a primary emphasis on fine-tuning the T5 model and exploring the capabilities of GPT-3.5 Turbo and GPT-4 in zero-shot and one-shot settings.

## Overview

Entity relation extraction plays a crucial role in natural language processing, and while traditional approaches concentrate on sentence-level analysis, this research delves into the emerging domain of document-level entity relation extraction. The complexity arises from dependencies and connections that go beyond individual sentences, presenting unique challenges in understanding relationships within a broader context.

## Key Features

- Exploration of generative models for document-level entity relation extraction.
- Fine-tuning of the T5 model and its variants for capturing nuanced relationships.
- Evaluation of GPT-3.5 Turbo and GPT-4 in zero-shot and one-shot scenarios.
- Comparative analysis of T5 and GPT models for document-level extraction performance.

## How to Use

1. **Clone the repository:**

    ```bash
    git clone https://github.com/dunja274/relation-extraction.git
    cd entity-relation-extraction
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run experiments and evaluate models using provided scripts.**

## Requesting Full Thesis and Results

This repository presents a condensed version of the thesis. For a detailed understanding of the research methodology, results, and conclusions, you can request the full thesis by sending an email to [your@email.com](mailto:dunja.smigovec@gmail.com). 

## Thesis Structure

- **Chapter 2**: Review of related work in entity relation extraction.
- **Chapter 3**: Insights into the utilized dataset.
- **Chapter 4**: Task modeling, experimental setup, and prompt engineering.
- **Chapter 5**: Discussion of models, including T5 variants and GPT models.
- **Chapter 6**: Evaluation metrics, including ROUGE-1, ROUGE-2, and Rouge-L scores.
- **Chapter 7**: Analysis of experiments and model performances.
- **Chapter 8**: Conclusions, limitations, and suggestions for future work.

**Note:** This project is a part of an academic thesis, and the full thesis can be requested for a more in-depth exploration of the research and its results.

*Keywords: Natural Language Processing, Entity Relation Extraction, Document-Level Analysis, T5 Model, GPT Models, Generative Approaches, Information Extraction.*


# Thesis Outline

1. **Introduction**
   - Brief overview of the research topic.

2. **Related Work**
   2.1. Early Approaches  
      - Discussion of early approaches in entity relation extraction.
   2.2. Deep Learning Approaches  
      - Overview of deep learning approaches in entity relation extraction.

3. **Dataset**
   3.1. Preprocessing  
      - Details on dataset preprocessing.
   3.2. Statistics  
      - Presentation of dataset statistics.

4. **Task Modeling**
   4.1. Relation Extraction as Extractive Summarization  
      - Exploration of relation extraction as extractive summarization.
   4.2. Experiments  
      4.2.1. Named Entity Recognition and Relation Extraction  
         - Details on experiments involving named entity recognition and relation extraction.
      4.2.2. Tags and Relation Extraction  
         - Overview of experiments involving tags and relation extraction.
      4.2.3. Entities and Relation Extraction  
         - Experiments related to entities and relation extraction.
   4.3. Prompting  
      4.3.1. Named Entity Recognition and Relation Extraction Prompt Design  
         - Design of prompts for named entity recognition and relation extraction.
      4.3.2. Relation Extraction Prompt Design  
         - Design of prompts for relation extraction.
      4.3.3. Entities and Relation Extraction Prompt Design  
         - Design of prompts for entities and relation extraction.

5. **Models**
   5.1. Text-to-Text Transfer Transformers  
      5.1.1. Multilingual T5  
         - Discussion on multilingual T5.
      5.1.2. FLAN-T5  
         - Overview of FLAN-T5.
   5.2. Generative Pre-trained Transformers  
      5.2.1. GPT-3.5 Turbo  
         - Details on GPT-3.5 Turbo.
      5.2.2. GPT-4  
         - Overview of GPT-4.
   5.3. Comparison of T5 and GPT models  
      - Comparative analysis of T5 and GPT models.

6. **Metrics**
   6.1. ROUGE-1 Score  
      - Explanation of ROUGE-1 score.
   6.2. ROUGE-2 Score  
      - Explanation of ROUGE-2 score.
   6.3. ROUGE-L Score  
      - Explanation of ROUGE-L score.
   6.4. Comparison of ROUGE Scores  
      - Comparative analysis of ROUGE scores.
   6.5. Comparison Between ROUGE and other Similarity Scores  
      - Comparative analysis between ROUGE and other similarity scores.

7. **Results and Discussion**
   7.1. T5 Results  
      7.1.1. Analysis of Model Performance and Experimental Configurations  
         - Discussion on T5 model performance and experimental configurations.
      7.1.2. Analysis of Model Performance and Dataset Size  
         - Discussion on T5 model performance concerning dataset size.
   7.2. GPT Results  
      - Discussion on GPT model results.

8. **Conclusions**

**Bibliography**

