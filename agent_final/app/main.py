from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

class FinalInput(BaseModel):
    analysis: str
class AgentOutput(BaseModel):
    output: str

app = FastAPI()
llm = ChatOllama(
    model="llama3", # NOTA: Usiamo llama3
    base_url=os.environ.get("OLLAMA_BASE_URL"),
    temperature=0.0
)

@app.post("/invoke", response_model=AgentOutput)
def invoke_agent(data: FinalInput):
    print("Agente Finale: Ricevuta richiesta...")
    prompt = ChatPromptTemplate.from_template(
        """Sei l'agente finale.
        Basandoti sulla seguente analisi:
        {analysis}
        
        Qual è la risposta finale corretta tra le scelte (A), (B), (C), (D)?
        Rispondi solo con la lettera e il valore (es. (X) 10^-Y eV)."""
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"analysis": data.analysis})
    return {"output": result}