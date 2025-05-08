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
    prompt_id = 1
    arr = [(0, 4),(4, 8), (8, 11)]
    for i in range(len(arr)):
    
        rubrics, submissions = load_data(prompt_id=prompt_id)
        submissions = submissions[arr[i][0]:arr[i][1]]
        input = {
            'rubrics': rubrics,
            'submissions': submissions,
            'max_number_of_retries': 1,
            'output_folder': "gradedData",
            'grader_agent_file_name': os.environ.get("OUTPUT_FILE"),
            'agent_categories':  ['EssayID','PromptID','Content','Adherence','Language','PromptAdherence','Narrativity', 'Feedback'] 
            if prompt_id > 2 else ['EssayID','PromptID','Content', 'Organization','WordChoice','SentenceFluency', 'Conventions',"Feedback"],
            'no_of_grader_agents': 3
        }
        try:
            #write to timeLog.tdt
            with open(f"timeLog.txt_{prompt_id}", "a") as f:
                # write count of submissions
                f.write(f"Number of submissions: {len(submissions)}\n")
                # write prompt id
                f.write(f"Prompt ID: {prompt_id}\n")
                f.write(f"Start time: {datetime.now().isoformat()}\n")
                start = time.time()
                CrewAgents().crew().kickoff(inputs=input)
                end = time.time()
                f.write(f"End time: {datetime.now().isoformat()}\n")
                f.write(f"Execution time: {end - start} seconds\n")
        except Exception as e:
            raise Exception(f"An error occurred while running the crew: {e}")

    
   

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
    

def load_data(prompt_id):
    print("Loading ASAP++ dataset...")
    
    guidelines_dir = 'C:\\Users\\NOBZ\\source\\repos\\thesis\\crew_agents\\prompts\\guidelines'
    submissions_file = f"C:/Users/NOBZ/source/repos/thesis/crew_agents/submissions/training_data_{prompt_id}.csv"
    
    # Initialize lists to store results
    rubric_list = []
    submission_list = []
    
    # Load rubrics
    prompt_folder = os.path.join(guidelines_dir, f'prompt{prompt_id}')
    rubric_json_file = os.path.join(prompt_folder, f'rubric.json')

    if os.path.exists(rubric_json_file):
        with open(rubric_json_file, 'r') as f:
            rubric_data = json.load(f)
            rubric_list.append(rubric_data)  # Add to rubric list
            rubric_df = pd.json_normalize(rubric_data)
            print(f"📄 Prompt-{prompt_id} rubric.json head:")
            print(rubric_df.head())

    # Load submissions
    submissions_df = pd.read_csv(submissions_file)
    print(submissions_df.head())
    submission_list = submissions_df.to_dict(orient='records')
    
    print(f"✅ Loaded {len(submission_list)} submissions")
    print("Sample submission:")
    print(submission_list[0] if submission_list else "No submissions loaded")
    
    return rubric_list, submission_list
