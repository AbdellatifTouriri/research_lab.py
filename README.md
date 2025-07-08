# research_lab.py
#  AGNO Research Lab - Multi-Agent AI System

This is a fully autonomous multi-agent system built with [AGNO](https://github.com/multion/agnos), designed to:

- Read and summarize scientific PDFs
- Scrape relevant websites
- Analyze and critique research
- Propose innovative hypotheses
- Plan experiments
- Simulate peer review
- Generate grant proposals

##   Features

-  Real PDF integration (via PyMuPDF)
-  Website scraping (via BeautifulSoup)
-  Custom vector memory using SentenceTransformers
-  Modular agents with memory and toolkits
-  Optional Streamlit or FastAPI support

##  Run It

```bash
pip install -r requirements.txt
python agno_research_lab.py
