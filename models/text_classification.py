# models/text_classification.py
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

class ToxicityClassifier:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        config = PeftConfig.from_pretrained(model_path)
        with open(f"{model_path}/label_mappings.json", "r", encoding="utf-8") as f:
            label_data = json.load(f)
            self.id2label = label_data["id2label"]
            self.label2id = label_data["label2id"]
            num_labels = len(self.id2label)

        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model = PeftModel.from_pretrained(base_model, model_path).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def predict(self, text: str) -> dict:
        """Return dict with predicted_class and confidence_score (float)."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=500,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        confidence, class_id = torch.max(probs, dim=1)
        return {
            "predicted_class": self.id2label[str(class_id.item())],
            "confidence_score": float(confidence.item())
        }
