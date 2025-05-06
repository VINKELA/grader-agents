from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crew_agents.tools.custom_tool import FileWriterTool, ReadFileTool, FinalFileWriter,ComprehensiveGradingAnalyzer


@CrewBase
class CrewAgents():
    """CrewAgents crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    grader_llm: str = "gpt-3.5-turbo" 
    cordinator_llm: str = "gpt-4o"
    reflector_llm: str = "gpt-3.5-turbo"   

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
   
    @agent
    def grader(self) -> Agent:
        return Agent(
            config=self.agents_config['grader'], # type: ignore[index]
            verbose=True,
            llm=self.grader_llm
        )

    @agent
    def cordinator(self) -> Agent:
        return Agent(
            config=self.agents_config['cordinator'], # type: ignore[index]
            verbose=True,
            llm=self.cordinator_llm,
            tools=[
                FileWriterTool(),
                ReadFileTool(),
                ComprehensiveGradingAnalyzer()                
            ]
            
        )
    @agent
    def reflector(self) -> Agent:
        return Agent(
            config=self.agents_config['reflector'], # type: ignore[index]
            verbose=True,
            tools=[
                ReadFileTool() 
            ],
            llm=self.reflector_llm
            
        )
    
    @task
    def grading(self) -> Task:
        return Task(
            config=self.tasks_config['grading'] # type: ignore[index]
        )
    @task
    def cordination(self) -> Task:
        return Task(
            config=self.tasks_config['cordination'],
            output_file= "log.txt",
        )
    @task
    def reflection(self) -> Task:
        return Task(
            config=self.tasks_config['reflection']
        )
    @crew
    def crew(self) -> Crew:
        """Creates the CrewAgents crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
