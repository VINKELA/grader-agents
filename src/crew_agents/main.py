#!/usr/bin/env python
import sys
import warnings

from datetime import datetime
import os
import json
import pandas as pd

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
    submissions = submissions[1:4]
    
    input = {
        'rubrics': rubrics,
        'submissions': submissions,
        'max_number_of_retries': 1,
        'output_folder': "gradedData",
        'log_file': "log.txt",
        'summary_file_name': "grades.csv",
        'grader_agent_file_name': "grader_agent.csv",
        'summary_categories': ['EssayID','PromptID','AverageContent', 'AverageOrganization', 'AverageWordChoice', 'AverageSentenceFluency', 'AverageConventions', 
                                 "FeedbackSimilarityScore", "AverageFeedbackAdherenceToRubrics"],
        'agent_categories': ['EssayID','PromptID','Content', 'Organization', 'WordChoice', 'SentenceFluency', 'Conventions', 
                               "Feedback"],
        'no_of_grader_agents': 3
    }
    try:
        CrewAgents().crew().kickoff(inputs=input)
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
    

def load_data():
    print("Loading ASAP++ dataset...")
    
    guidelines_dir = 'C:\\Users\\NOBZ\\source\\repos\\thesis\\crew_agents\\src\\crew_agents\\data\\guidelines'
    submissions_file = "C:/Users/NOBZ/source/repos/thesis/crew_agents/src/crew_agents/data/submissions/training_set.csv"
    
    # Initialize lists to store results
    rubric_list = []
    submission_list = []
    
    # Load rubrics
    for i in range(1, 2):
        prompt_folder = os.path.join(guidelines_dir, f'prompt{i}')
        rubric_json_file = os.path.join(prompt_folder, f'rubric.json')

        if os.path.exists(rubric_json_file):
            with open(rubric_json_file, 'r') as f:
                rubric_data = json.load(f)
                rubric_list.append(rubric_data)  # Add to rubric list
                rubric_df = pd.json_normalize(rubric_data)
                print(f"📄 Prompt-{i} rubric.json head:")
                print(rubric_df.head())

    # Load submissions
    submissions_df = pd.read_csv(submissions_file)
    submission_list = submissions_df.to_dict(orient='records')
    
    print(f"✅ Loaded {len(submission_list)} submissions")
    print("Sample submission:")
    print(submission_list[0] if submission_list else "No submissions loaded")
    
    return rubric_list, submission_list
