import json
from crewai.tools import BaseTool
from typing import Type, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import os
import csv

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
