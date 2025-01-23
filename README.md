```markdown
# M-BEHRT: A Multimodal BERT Model for Electronic Health Records

M-BEHRT is an adaptation of the BERT architecture tailored for multimodal electronic health records (EHRs). Designed to process and integrate information from diverse clinical data modalities, M-BEHRT leverages the complementary strengths of **free-text reports**, **biological features**, and **clinical descriptors** to provide a comprehensive patient representation.

M-BEHRT has been applied to a **breast cancer cohort** from the **Institut Curie**, demonstrating its utility in predicting **disease-free survival (DFS)** at **3 and 5 years** after the first surgery.

The model is built as a combination of two specialized BEHRT models:

### 1. **Text BEHRT**
This component is optimized for processing **free-text reports**. Free-text reports often contain rich, unstructured narratives from healthcare providers, such as progress notes, discharge summaries, and radiology interpretations. 

### 2. **Tabular BEHRT**
This component is tailored for processing **structured, tabular data** commonly found in EHRs, such as time-stamped biological features and categorical clinical descriptors (e.g., demographic data etc.).

## Multimodal Fusion

The integration of Text BEHRT and Tabular BEHRT enables M-BEHRT to effectively combine information from different modalities. 

## Advantages of M-BEHRT

- **Holistic Patient Representation**: By combining free-text and structured data, M-BEHRT provides a more comprehensive view of patient health.
- **Enhanced Predictive Performance**: The model's ability to leverage both modalities results in improved performance on tasks such as disease prediction, risk stratification, and treatment recommendation.
- **Scalability**: M-BEHRT can be fine-tuned for specific healthcare applications, making it a versatile tool for clinical and research settings.
