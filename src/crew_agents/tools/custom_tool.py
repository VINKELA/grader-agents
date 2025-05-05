import json
from crewai.tools import BaseTool
from typing import Type, List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field

import matplotlib.pyplot as plt
import os

import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

class CalculateMeanScoresInput(BaseModel):
    """Input schema for CalculateMeanScoresTool."""
    scores_json: str = Field(description="A JSON string containing  scores and feedbacks for a submission."),
    rubric_keywords: List[str] = Field(description="List of rubric keywords.")

class CalculateMeanScoresTool(BaseTool):
    name: str ="calculate_mean_scores"
    description: str = (
        "Calculates the mean score across four grader agents' outputs "
        "and computes the feedback adherence to rubrics"
    )
    args_schema: Type[BaseModel]  = CalculateMeanScoresInput

    def _run(self, scores_json: str, rubric_keywords:List[str]) -> str:
        data = json.loads(scores_json)
        categories = ['AverageContent', 'AverageOrganization', 'AverageWordChoice', 'AverageSentenceFluency', 'AverageConventions' 
                            ]
        mean_scores = {}

        for category in categories:
            mean_scores[category] = sum(agent[category] for agent in data) / len(data)
        #get feedback list from all agents
        feedbacks = [agent['Feedback'] for agent in data]
        #caluculate the average feedback adherence to rubrics
        adherences = [calculate_rubrics_adherence(rubric_keywords, feedback) for feedback in feedbacks]
        mean_scores["AverageFeedbackAdherenceToRubrics"] = sum(float(ad) for ad in adherences) / len(adherences)
        #calcuate feedback similarity score
        vectorizer = TfidfVectorizer().fit_transform(feedbacks)
        result = {**mean_scores, 'FeedbackSimilarityScore': vectorizer}
        return json.dumps(result)

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
        


def calculate_rubrics_adherence(rubric_keywords: list, feedback: str) -> str:
    count = sum(1 for kw in rubric_keywords if kw.lower() in feedback.lower())
    score = (count / len(rubric_keywords)) * 100 if rubric_keywords else 0
    return str(score)


class FeedbackSimilarityInput(BaseModel):
    """Input schema for FeedbackSimilarityTool."""
    feedback_list: list = Field(..., description="List of feedback strings from graders.")

class FeedbackSimilarityTool(BaseTool):
    name: str = "calculate_feedback_similarity"
    description: str = (
        "Calculates similarity score between multiple feedbacks using cosine similarity over TF-IDF vectors."
    )
    args_schema: Type[BaseModel] = FeedbackSimilarityInput

    def _run(self, feedback_list: list) -> str:
        if len(feedback_list) < 2:
            return "100.0"  # Perfect similarity if only one input
        vectorizer = TfidfVectorizer().fit_transform(feedback_list)
        similarity_matrix = cosine_similarity(vectorizer)
        n = len(feedback_list)
        total_sim = sum(
            similarity_matrix[i, j]
            for i in range(n) for j in range(i + 1, n)
        )
        avg_sim = total_sim / (n * (n - 1) / 2)
        return str(avg_sim * 100)
