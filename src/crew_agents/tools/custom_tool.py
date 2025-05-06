from collections import defaultdict
import json
from crewai.tools import BaseTool
from typing import Type, List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, PrivateAttr
import os
import csv
from sentence_transformers import SentenceTransformer
import numpy as np
from torch import cosine_similarity

class FileWriterToolInput(BaseModel):
    """Input schema for FileWriterTool."""
    data: List[dict] = Field(description="List of grading records to save")
    file_path: str = Field(description="Output file path (auto-appends .csv if missing)")
    mode: str = Field(default="w", description="'w' (overwrite) or 'a' (append)")
    fieldnames: Optional[List[str]] = Field(
        default=None,
        description="field names for CSV"
    )

class FileWriterTool(BaseTool):
    name: str = "File Writer"
    description: str = (
        "Writes essay grading data to CSV files with standardized fields. "
    )
    args_schema: Type[BaseModel] = FileWriterToolInput

    def _run(
        self,
        data: List[dict],
        file_path: str,
        mode: str = "w",
        fieldnames: Optional[List[str]] = None
    ) -> str:
        # Set default fields for essay grading
        fields = fieldnames
        try:
            # Ensure CSV extension
            if not file_path.lower().endswith('.csv'):
                file_path = f"{file_path}.csv"
            

            # Prepare data (ensure all fields exist)
            clean_data = []
            for record in data:
                clean_record = {field: record.get(field, "") for field in fields}
                clean_data.append(clean_record)

            # Write CSV
            file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
            with open(file_path, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                if not file_exists or mode == 'w':
                    writer.writeheader()
                writer.writerows(clean_data)

            return json.dumps({
                "status": "success",
                "path": os.path.abspath(file_path),
                "records": len(clean_data),
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

class GradingAnalysisInput(BaseModel):
    """Input schema for ComprehensiveGradingAnalyzer."""
    grading_data: List[Dict] = Field(..., description="List of raw grading records from all agents")
    numeric_fields: List[str] = Field(
        default=["Content", "Organization", "WordChoice", "SentenceFluency", "Conventions"],
        description="Numeric columns to average"
    )
    rubric_keywords: List[str] = Field(
        default=["content", "organization", "word choice", "sentence fluency", "conventions"],
        description="Keywords from rubric for feedback analysis"
    )

class ComprehensiveGradingAnalyzer(BaseTool):
    name: str = "comprehensive_grading_analyzer"
    description: str = (
        "Performs complete grading analysis including score averaging, "
        "feedback similarity measurement, and rubric adherence scoring. "
        "Returns consolidated results for each submission."
    )
    args_schema: Type[BaseModel] = GradingAnalysisInput
    _model: SentenceTransformer = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = SentenceTransformer('all-MiniLM-L6-v2')

    def _calculate_similarity(self, feedbacks: List[str]) -> float:
        """Calculate cosine similarity between multiple feedback texts"""
        if len(feedbacks) < 2:
            return 1.0  # Perfect similarity if only one feedback
            
        embeddings = np.array(self._model.encode(feedbacks, convert_to_tensor=False))
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
            
        similarity_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(similarity_matrix, 0)
        return float(np.mean(similarity_matrix))

    def _calculate_adherence(self, feedback: str, keywords: List[str]) -> float:
        """Calculate percentage of rubric keywords covered in feedback"""
        if not feedback or not keywords:
            return 0.0
        feedback_lower = feedback.lower()
        matches = sum(keyword.lower() in feedback_lower for keyword in keywords)
        return (matches / len(keywords)) * 100 if keywords else 0.0

    def _run(self, grading_data: List[Dict], numeric_fields: List[str], rubric_keywords: List[str]) -> List[Dict]:
        # Group submissions by EssayID and PromptID
        grouped = defaultdict(list)
        for record in grading_data:
            if not isinstance(record, dict):
                continue
            try:
                key = (int(record.get("EssayID", 0)), int(record.get("PromptID", 0)))
                grouped[key].append(record)
            except (ValueError, TypeError):
                continue

        results = []
        for (essay_id, prompt_id), records in grouped.items():
            if not records:
                continue

            # Calculate average scores
            avg_record = {
                "EssayID": essay_id,
                "PromptID": prompt_id
            }
            
            for field in numeric_fields:
                values = [float(r[field]) for r in records if field in r and isinstance(r[field], (int, float))]
                avg_record[f"Average{field}"] = round(np.mean(values), 2) if values else 0.0

            # Analyze feedback
            feedbacks = [r["Feedback"] for r in records if isinstance(r.get("Feedback"), str)]
            
            similarity = self._calculate_similarity(feedbacks) if feedbacks else 0.0
            adherence_scores = [
                self._calculate_adherence(fb, rubric_keywords) 
                for fb in feedbacks
                if isinstance(fb, str)
            ]
            avg_adherence = np.mean(adherence_scores) if adherence_scores else 0.0

            avg_record.update({
                "FeedbackSimilarityScore": round(similarity, 2),
                "AverageFeedbackAdherenceToRubrics": round(avg_adherence, 2)
            })
            
            results.append(avg_record)
            
        return results
class ReadFileInput(BaseModel):
    """Input schema for ReadFileTool."""
    file_path: str = Field(description="The full path to the grader.csv file to read and review.")

class ReadFileTool(BaseTool):
    name: str ="read_csv_file"
    description: str = (
        "Reads the content of the specified grader.csv file so the reflector agent can review it for format, "
        "required headers, and proper CSV structure."
    )
    args_schema: Type[BaseModel]  = ReadFileInput

    def _run(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error reading file: {e}"

# ======================
# 3. FINAL FILE WRITER
# ======================
class FinalFileWriterInput(BaseModel):
    """Input schema for FinalFileWriter."""
    data: List[Dict] = Field(..., description="List of processed grading records to save")
    file_path: str = Field(..., description="Output file path (auto-appends .csv if missing)")
    mode: str = Field(default="w", description="File mode: 'w' (write) or 'a' (append)")

class FinalFileWriter(BaseTool):
    name: str = "final_grades_writer"
    description: str = (
        "Writes final aggregated grades with feedback analysis to CSV. "
        "Ensures standardized output format with required fields."
    )
    args_schema: Type[BaseModel] = FinalFileWriterInput

    def _run(self, data: List[Dict], file_path: str, mode: str = "w") -> str:
        required_fields = [
            "EssayID", "PromptID", 
            "AverageContent", "AverageOrganization", 
            "AverageWordChoice", "AverageSentenceFluency",
            "AverageConventions", "FeedbackSimilarityScore",
            "AverageFeedbackAdherenceToRubrics"
        ]
        
        if not file_path.lower().endswith('.csv'):
            file_path = f"{file_path}.csv"
        
        try:
            clean_data = []
            for record in data:
                clean_record = {field: record.get(field, "") for field in required_fields}
                clean_data.append(clean_record)

            file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
            with open(file_path, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=required_fields)
                if not file_exists or mode == 'w':
                    writer.writeheader()
                writer.writerows(clean_data)

            return json.dumps({
                "status": "success",
                "path": os.path.abspath(file_path),
                "records": len(clean_data),
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })