# Toxic Content Classification

This repository contains a multi-part NLP/AI training project for toxic content classification, including a Streamlit web application for text and image-based toxicity prediction.

## Repository Structure

- `Task-0/`
  - `Task 0.pdf` - Task 0 documentation or assignment description.
- `Task1/` - Main application project for toxic content classification.

## Main Folder Overview

`Main/` contains the main Streamlit app and supporting assets:

- `app.py` - Streamlit application entrypoint.
- `combined_data.csv` - CSV data store for classification records.
- `data_base/`
  - `database.py` - Simple CSV-backed data persistence layer.
- `models/`
  - `imagecaption.py` - Image caption generation using BLIP.
  - `text_classification.py` - Toxicity classifier using a PEFT-enabled DistilBERT model.
  - `Distil-BERT_model/` - Pretrained model files and adapter weights.
    - `adapter_config.json`
    - `adapter_model.safetensors`
    - `label_mappings.json`
    - `README.md`
    - `special_tokens_map.json`
    - `tokenizer_config.json`
    - `tokenizer.json`
    - `vocab.txt`
    - `lora_distilbert_toxic/` - Nested adapter/model files.
- `uploads/` - Directory where uploaded images are stored.
- `README.md` - README.

## What This Project Does

The application provides:

- Text toxicity classification
- Image caption generation
- Toxicity classification of generated captions from uploaded images
- Automatic record persistence in a CSV file
- Streamlit-based UI with tabs for input and database review

## How It Works

- `app.py` launches a Streamlit web application.
- `models/imagecaption.py` loads `Salesforce/blip-image-captioning-base` and generates captions for image inputs.
- `models/text_classification.py` loads a PEFT adapter model from `models/Distil-BERT_model` and classifies text.
- `data_base/database.py` writes classification records into `combined_data.csv` and can display stored results.

## Setup Instructions

1. Open a terminal in the root workspace folder:

   ```powershell
   cd "d:Task"
   ```

2. Create a Python virtual environment (recommended):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install required Python packages. This project depends on:

   - `streamlit`
   - `pandas`
   - `torch`
   - `transformers`
   - `peft`
   - `Pillow`
   - `requests`

   Install packages with:

   ```powershell
   pip install streamlit pandas torch transformers peft pillow requests
   ```

4. Ensure the model path in `Task/app.py` is correct for your environment. It is currently set to:

   ```python
   TOX_MODEL_PATH = r"D:\Task\models\Distil-BERT_model"
   ```

   Update this path if you move the repository or run it from another machine.

## Running the Streamlit App

From `Task/`:

```powershell
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Usage Notes

- Use the **Text** input tab to classify free-form text.
- Use the **Image** input tab to upload an image, generate a caption, and classify that caption.
- Records are saved in `combined_data.csv` and displayed in the **View Database** tab.
- Uploaded images are stored in the `uploads/` directory.

## Notes on Data Storage

- `combined_data.csv` is automatically created if it does not exist.
- `data_base/database.py` writes rows with the following columns:
  - `original_input`
  - `caption`
  - `predicted_class`
  - `confidence`

## Important Considerations

- The project currently uses an absolute model path in `Task/app.py`. It is best to update this to a relative path or environment-specific configuration for portability.
- The `models/Distil-BERT_model` directory contains the fine-tuned text classification model and associated tokenizer files.
- If `torch.cuda.is_available()` returns `True`, the app uses GPU acceleration; otherwise it runs on CPU.

## Recommended Improvements

- Add a `requirements.txt` file for reproducible dependency installation.
- Convert the hard-coded `TOX_MODEL_PATH` into a configuration value or environment variable.
- Add error handling around model loading and prediction failures.
- Document the specific toxicity labels used by `label_mappings.json`.

## Task-0 Assets

The root folder also contains Task 0 resources:

- `Task 0.pdf`

These files appear to be separate task documentation or deliverables and are not part of the `Task` application.

---

