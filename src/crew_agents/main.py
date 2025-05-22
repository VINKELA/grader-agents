#!/usr/bin/env python
import sys
import warnings

from datetime import datetime
import os
import json
import pandas as pd
import time
from crew_agents.crew import CrewAgents

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    rubrics, submissions = load_data()
    for i in range(len(submissions)):
        input = {
            'rubrics': rubrics,
            'submissions': submissions[i],
            'max_number_of_retries': 1,
            'output_folder': os.environ.get("OUTPUT_FOLDER"),
            'grader_agent_file_name': os.environ.get("OUTPUT_FILE"),
            'agent_categories':  [os.environ.get("OUTPUT_FILE_HEADERS")], 
            'no_of_grader_agents': 3
        }
        try:
            #write to timeLog.tdt
            log_file = os.environ.get("TIME_LOG_FILE")
            with open(log_file, "a") as f:
                # write count of submissions
                f.write(f"Number of submissions: {1}\n")
                # write prompt id
                f.write(f"Start time: {datetime.now().isoformat()}\n")
                start = time.time()
                CrewAgents().crew().kickoff(inputs=input)
                end = time.time()
                f.write(f"End time: {datetime.now().isoformat()}\n")
                f.write(f"Execution time: {end - start} seconds\n")
        except Exception as e:
            raise Exception(f"An error occurred while running the crew: {e}")
        time.sleep(10)  # Sleep for 1 second between each submission

    
   

def train():
    """
    Train the crew for a given number of iterations.
    """
    rubrics, submissions = load_data()
    inputs = {
        'rubrics': rubrics,
        'submissions': submissions,
    }
    try:
        CrewAgents().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        CrewAgents().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
    }
    
    try:
        CrewAgents().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
    

def load_data():
    print("Loading ASAP++ dataset...")
    
    rubric_dir = os.environ.get("RUBRICS_FOLDER")
    folder = os.environ.get("SUBMISSION_FOLDER")
    file = os.environ.get("SUBMISSION_FILE")
    #prompt_folder = os.path.join(rubric_dir, f'prompt{prompt_id}')
    rubric_json_file = os.path.join(rubric_dir, f'{os.environ.get("RUBRIC_FILE")}')

    submissions_file = os.path.join(folder, file)
    if not rubric_json_file or not submissions_file:
        raise ValueError("Please set the RUBRIC_FOLDER variable.")
    if not submissions_file:
        raise ValueError("Please set the SUBMISSION_FOLDER environment variables.")

    #validate promp

    # Initialize lists to store results
    rubric_list = []
    submission_list = []
    output_folder = os.environ.get("OUTPUT_FOLDER")
    output_file = os.environ.get("OUTPUT_FILE")
    output_file = os.path.join(output_folder, output_file)
    #check if output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    #check if output file is a csv file
    if not output_file.endswith('.csv'):
        raise ValueError("Output file must be a CSV file")
    if not rubric_json_file.endswith('.json'):
        raise ValueError("Rubric file must be a JSON file")
    
    #check if submission file is a csv file
    if not submissions_file.endswith('.csv'):
        raise ValueError("Submission file must be a CSV file")
    
    #check if output file exists
    if not os.path.exists(output_file):
        #create the file
        with open(output_file, 'w') as f:
            #write the header
            f.write(os.environ.get("OUTPUT_FILE_HEADERS"))

    # Load rubrics
    if os.path.exists(rubric_json_file):
        with open(rubric_json_file, 'r') as f:
            rubric_data = json.load(f)
            rubric_list.append(rubric_data)  # Add to rubric list
            rubric_df = pd.json_normalize(rubric_data)
            print(f"📄 rubric head:")
            print(rubric_df.head())
    else:
        raise FileNotFoundError(f"Rubric file {rubric_json_file} not found.")

    # Load submissions
    submissions_df = pd.read_csv(submissions_file)
    print(submissions_df.head())
    submission_list = submissions_df.to_dict(orient='records')
    
    print(f"✅ Loaded {len(submission_list)} submissions")
    print("Sample submission:")
    print(submission_list[0] if submission_list else "No submissions loaded")
    
    return rubric_list, submission_list
