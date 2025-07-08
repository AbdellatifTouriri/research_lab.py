 

from agno.agent import Agent
from agno.team import AgentTeam
from agno.memory.shared import SharedMemory
from agno.models.openai import OpenAIChat
from agno.memory.vector import VectorMemory
from agno.knowledge.base import StaticKnowledge
from agno.utils.pprint import pprint_run_response

from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import requests
import fitz   
class PdfReaderTool:
    def run(self, url):
        response = requests.get(url)
        with open("temp.pdf", "wb") as f:
            f.write(response.content)

        doc = fitz.open("temp.pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text[:3000]   
    
class WebScraperTool:
    def run(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        return text[:3000]
 
class CustomEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text):
        return self.model.encode([text])[0]  

class CustomVectorMemory(VectorMemory):
    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder

    def embed(self, text):
        return self.embedder.embed(text)
 
embedder = CustomEmbedder()
vector_memory = CustomVectorMemory(embedder)
shared_memory = SharedMemory()

knowledge = StaticKnowledge(content="""
- Scientific method fundamentals and experimental design best practices.
- Key AI breakthroughs: transformers, reinforcement learning, self-supervised learning.
- Research paper structure: abstract, introduction, methods, results, discussion.
- Zero-shot learning techniques and challenges.
""")

crawler_agent = Agent(
    name="CrawlerAgent",
    goal="Fetch research papers and datasets.",
    model=OpenAIChat(id="gpt-4o-mini"),
    memory=shared_memory,
    knowledge=knowledge,
    tools=[WebScraperTool(), PdfReaderTool()]
)

semantic_indexer_agent = Agent(
    name="SemanticIndexerAgent",
    goal="Embed and cluster documents.",
    model=OpenAIChat(id="gpt-4o-mini"),
    memory=vector_memory,
    knowledge=knowledge,
)

summarizer_agent = Agent(
    name="SummarizerAgent",
    goal="Summarize key research findings.",
    model=OpenAIChat(id="gpt-4o-mini"),
    memory=vector_memory,
    knowledge=knowledge,
)

critique_agent = Agent(
    name="CritiqueAgent",
    goal="Analyze research critically.",
    model=OpenAIChat(id="gpt-4o-mini"),
    memory=shared_memory,
    knowledge=knowledge,
)

innovation_agent = Agent(
    name="InnovationAgent",
    goal="Propose research directions.",
    model=OpenAIChat(id="gpt-4o-mini"),
    memory=shared_memory,
    knowledge=knowledge,
)

experiment_planner_agent = Agent(
    name="ExperimentPlannerAgent",
    goal="Design experiments.",
    model=OpenAIChat(id="gpt-4o-mini"),
    memory=shared_memory,
    knowledge=knowledge,
)

grant_writer_agent = Agent(
    name="GrantWriterAgent",
    goal="Write grant proposals.",
    model=OpenAIChat(id="gpt-4o-mini"),
    memory=shared_memory,
    knowledge=knowledge,
)

peer_review_simulator_agent = Agent(
    name="PeerReviewSimulatorAgent",
    goal="Simulate peer-review feedback.",
    model=OpenAIChat(id="gpt-4o-mini"),
    memory=shared_memory,
    knowledge=knowledge,
)

leader_agent = Agent(
    name="LeaderAgent",
    goal="Coordinate team and integrate results.",
    model=OpenAIChat(id="gpt-4o-mini"),
    memory=shared_memory,
    knowledge=knowledge,
)

team = AgentTeam(
    agents=[
        crawler_agent,
        semantic_indexer_agent,
        summarizer_agent,
        critique_agent,
        innovation_agent,
        experiment_planner_agent,
        grant_writer_agent,
        peer_review_simulator_agent,
    ],
    leader=leader_agent,
    memory=shared_memory,
    knowledge=knowledge,
)

if __name__ == "__main__":
    print("\n--- Running AGNO Research Lab Assistant ---\n")
    response = team.run(
        "Conduct an advanced research project on 'Zero-shot learning in NLP'. "
        "Fetch papers, summarize, critique, innovate, plan experiments, write grant proposal, simulate peer-review.",
        stream=True,
        stream_intermediate_steps=True
    )
    pprint_run_response(response)

    print("\n--- Agent Events Log ---\n")
    for event in response.events:
        print(f"[{event.agent}] {event.event}")
