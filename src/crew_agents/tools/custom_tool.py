import json
from crewai.tools import BaseTool
from typing import Type, List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PyPDF2 import PdfReader
import docx
import csv
class FileWriterToolInput(BaseModel):
    """Input schema for FileWriterTool."""
    data: List[dict] = Field(description="List of grading records to save")
    file_path: str = Field(description="Output file path (auto-appends .csv if missing)")
    mode: str = Field(default="w", description="'w' (overwrite) or 'a' (append)")
    fieldnames: Optional[List[str]] = Field(
        default=None,
        description="Optional custom field names for CSV"
    )

class FileWriterTool(BaseTool):
    name: str = "File Writer"
    description: str = (
        "Writes essay grading data to CSV files with standardized fields. "
        "Default fields: EssayID, ContentScore, Organization, WordChoice, "
        "SentenceFluency, Conventions, Feedback, TotalScore"
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
        default_fields = [
            'EssayID', 'ContentScore', 'Organization', 'WordChoice',
            'SentenceFluency', 'Conventions', 'Feedback', 'TotalScore'
        ]
        fields = fieldnames or default_fields

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

class CalculateMeanScoresInput(BaseModel):
    """Input schema for CalculateMeanScoresTool."""
    scores_json: str = Field(description="A JSON string containing four sets of scores and feedbacks for a submission.")

class CalculateMeanScoresTool(BaseTool):
    name: str ="calculate_mean_scores"
    description: str = (
        "Calculates the mean score across four grader agents' outputs "
        "and selects one representative feedback (for now, picks the first)."
    )
    args_schema: Type[BaseModel]  = CalculateMeanScoresInput

    def _run(self, scores_json: str) -> str:
        data = json.loads(scores_json)
        categories = ['Content', 'Organization', 'Word Choice', 'Sentence Fluency', 'Conventions', 'Total Score']
        mean_scores = {}
        for category in categories:
            mean_scores[category] = sum(agent[category] for agent in data) / len(data)
        representative_feedback = data[0]['Feedback']
        result = {**mean_scores, 'Feedback': representative_feedback}
        return json.dumps(result)