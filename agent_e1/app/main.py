from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

class TaskInput(BaseModel):
    task: str
class AgentOutput(BaseModel):
    output: str

app = FastAPI()
llm = ChatOllama(
    model="llama3", # NOTA: Usiamo llama3
    base_url=os.environ.get("OLLAMA_BASE_URL"),
    temperature=0.0
)

@app.post("/invoke", response_model=AgentOutput)
def invoke_agent(data: TaskInput):
    print("Agente E1: Ricevuta richiesta...")
    prompt = ChatPromptTemplate.from_template(
        """Task: {task}
        Sub-task: Calcola l'incertezza energetica (Delta E) per il primo stato.
        Lifetime (Delta t) = 10^-9 sec.
        Usa hbar ≈ 6.582 x 10^-16 eV*s.
        Formula: Delta E ≈ hbar / Delta t
        Fornisci SOLO il calcolo e il risultato numerico in eV. Sii conciso."""
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"task": data.task})
    return {"output": result}