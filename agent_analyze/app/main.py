from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

class AnalysisInput(BaseModel):
    task: str
    e1_uncertainty: str
    e2_uncertainty: str
class AgentOutput(BaseModel):
    output: str

app = FastAPI()
llm = ChatOllama(
    model="llama3", # NOTA: Usiamo llama3
    base_url=os.environ.get("OLLAMA_BASE_URL"),
    temperature=0.0
)

@app.post("/invoke", response_model=AgentOutput)
def invoke_agent(data: AnalysisInput):
    print("Agente Analisi: Ricevuta richiesta...")
    prompt = ChatPromptTemplate.from_template(
        """Sei un analista di fisica.
        Task: {task}
        Opzioni: (A) 10^-9 eV, (B) 10^-11 eV, (C) 10^-8 eV, (D) 10^-4 eV
        
        Calcolo Incertezza 1: {e1_uncertainty}
        Calcolo Incertezza 2: {e2_uncertainty}
        
        Sub-task:
        1. Identifica quale incertezza (Delta E1 o Delta E2) è più grande. Questa è l'incertezza "dominante".
        2. Spiega che per 'risolvere chiaramente' i livelli, la differenza di energia tra loro deve essere *significativamente maggiore* dell'incertezza dominante.
        3. Confronta l'incertezza dominante con le opzioni (A, B, C, D).
        4. Scegli l'unica opzione che è *significativamente maggiore* dell'incertezza dominante. Motiva la scelta.
        """
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "task": data.task,
        "e1_uncertainty": data.e1_uncertainty,
        "e2_uncertainty": data.e2_uncertainty
    })
    return {"output": result}