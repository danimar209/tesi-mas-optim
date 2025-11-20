import requests
import json
import time
from typing import TypedDict, List, Literal

from langgraph.graph import StateGraph, END

# --- 1. DEFINIZIONE DEGLI URL DEI NOSTRI AGENTI (Microservizi) ---
AGENT_E1_URL = "http://agent_e1_service:8000/invoke"
AGENT_E2_URL = "http://agent_e2_service:8000/invoke"
AGENT_ANALYZE_URL = "http://agent_analyze_service:8000/invoke"
AGENT_FINAL_URL = "http://agent_final_service:8000/invoke"
OLLAMA_URL = "http://ollama_service:11434"

TASK = """Due stati quantistici con energie E1 ed E2 hanno una vita media di 10^-9 sec e 10^-8 sec, rispettivamente. 
Vogliamo distinguere chiaramente questi due livelli di energia. 
Quale delle seguenti opzioni potrebbe essere la loro differenza di energia in modo che possano essere chiaramente risolti?
Choices: (A) 10^-9 eV, (B) 10^-11 eV, (C) 10^-8 eV, (D) 10^-4 eV"""

# --- 2. DEFINIZIONE DELLO STATO (La "Memoria" del Sistema) ---
class GraphState(TypedDict):
    task: str
    e1_uncertainty: str
    e2_uncertainty: str
    analysis: str
    final_answer: str
    analysis_attempts: int
    max_attempts: int

# --- 3. DEFINIZIONE DEI NODI (I Nodi chiamano i Microservizi) ---

def node_calc_e1(state: GraphState) -> dict:
    print("-> NODO LANGGRAPH: Chiamata a agent_e1_service...", flush=True)
    payload = {"task": state['task']}
    response = requests.post(AGENT_E1_URL, json=payload)
    response.raise_for_status()
    result = response.json()['output']
    print(f"   Risposta E1 ricevuta.", flush=True)
    return {"e1_uncertainty": result}

def node_calc_e2(state: GraphState) -> dict:
    print("-> NODO LANGGRAPH: Chiamata a agent_e2_service...", flush=True)
    payload = {"task": state['task']}
    response = requests.post(AGENT_E2_URL, json=payload)
    response.raise_for_status()
    result = response.json()['output']
    print(f"   Risposta E2 ricevuta.", flush=True)
    return {"e2_uncertainty": result}

def node_analyze(state: GraphState) -> dict:
    print("-> NODO LANGGRAPH: Chiamata a agent_analyze_service...", flush=True)
    payload = {
        "task": state['task'],
        "e1_uncertainty": state['e1_uncertainty'],
        "e2_uncertainty": state['e2_uncertainty']
    }
    response = requests.post(AGENT_ANALYZE_URL, json=payload)
    response.raise_for_status()
    result = response.json()['output']
    print("   Analisi ricevuta.", flush=True)
    return {"analysis": result, "analysis_attempts": state['analysis_attempts'] + 1}

def node_final_answer(state: GraphState) -> dict:
    print("-> NODO LANGGRAPH: Chiamata a agent_final_service...", flush=True)
    payload = {"analysis": state['analysis']}
    response = requests.post(AGENT_FINAL_URL, json=payload)
    response.raise_for_status()
    result = response.json()['output']
    print(f"   Risposta Finale: {result.strip()}", flush=True)
    return {"final_answer": result}

# --- 4. DEFINIZIONE DELLA LOGICA "INTELLIGENTE" (Routing Condizionale) ---
# Questo è il cuore dell'ottimizzazione logica.
# Per la tesi, potresti sostituire questo con una chiamata a un "agente critico".

def node_should_continue(state: GraphState) -> Literal["continue", "end"]:
    print("-> NODO LANGGRAPH: Router (Controllo Qualità)", flush=True)
    if state['analysis_attempts'] >= state['max_attempts']:
        print("   Controllo: Tentativi massimi raggiunti. Fine.", flush=True)
        return "end"
    
    if not state['analysis'] or len(state['analysis']) < 50:
        print("   Controllo: Analisi fallita o troppo corta. Riprovo.", flush=True)
        return "continue" # CREA UN CICLO!
    else:
        print("   Controllo: Analisi OK. Proseguo.", flush=True)
        return "end"

# --- 5. FUNZIONI HELPER PER L'AVVIO ---

def pull_model(model_name: str):
    """Funzione helper robusta per scaricare il modello."""
    print(f"Avvio pull del modello '{model_name}' (potrebbe richiedere tempo)...", flush=True)
    try:
        with requests.post(f"{OLLAMA_URL}/api/pull", json={"name": model_name}, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if "status" in data:
                            status = data['status']
                            if "total" in data and "completed" in data:
                                print(f"Stato Ollama: {status} - {(data['completed'] / data['total'] * 100):.0f}%", end='\r', flush=True)
                            else:
                                print(f"\nStato Ollama: {status}", flush=True)
                            if status == "success":
                                print(f"\nModello '{model_name}' pronto.", flush=True)
                                return
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        print(f"Errore durante il pull: {e}. Il modello potrebbe essere già presente.", flush=True)
        
def wait_for_services():
    """Attende che tutti i servizi (Ollama e Agenti) siano pronti."""
    print("Attendo Ollama...", flush=True)
    retries = 10
    for _ in range(retries):
        try:
            requests.get(OLLAMA_URL, timeout=5)
            print("Ollama è pronto.", flush=True)
            break
        except requests.ConnectionError:
            time.sleep(5)
    else: raise Exception("Ollama non disponibile")
    
    print("Attendo gli Agenti...", flush=True)
    agent_urls = [AGENT_E1_URL, AGENT_E2_URL, AGENT_ANALYZE_URL, AGENT_FINAL_URL]
    for url in agent_urls:
        check_url = url.replace("/invoke", "/docs")
        for _ in range(retries):
            try:
                requests.get(check_url, timeout=5)
                print(f"Servizio {url} pronto.", flush=True)
                break
            except requests.ConnectionError:
                time.sleep(5)
        else: raise Exception(f"Agente {url} non disponibile")
    print("Tutti i servizi sono attivi.", flush=True)

# --- 6. COSTRUZIONE DEL GRAFO E AVVIO ---
if __name__ == "__main__":
    
    wait_for_services()
    pull_model("llama3") # Assicura che Llama 3 sia scaricato
    
    workflow = StateGraph(GraphState)

    workflow.add_node("calc_e1", node_calc_e1)
    workflow.add_node("calc_e2", node_calc_e2)
    workflow.add_node("analyze", node_analyze)
    workflow.add_node("final_answer", node_final_answer)

    # Nodi per l'ottimizzazione (logica condizionale)
    workflow.add_node("router", node_should_continue)

    # Flusso (E1 e E2 sono in parallelo? Per ora facciamoli sequenziali per semplicità)
    # Per la tesi, un'ottimizzazione è renderli paralleli!
    workflow.set_entry_point("calc_e1")
    workflow.add_edge("calc_e1", "calc_e2")
    workflow.add_edge("calc_e2", "analyze")
    
    # QUI C'È L'INTELLIGENZA:
    workflow.add_conditional_edges(
        "analyze",          # Nodo di partenza
        node_should_continue, # Funzione che decide la rotta
        {
            "continue": "analyze",      # Ritorna ad "analyze" (CICLO)
            "end": "final_answer"   # Prosegui
        }
    )
    workflow.add_edge("final_answer", END)

    app = workflow.compile()

    print("\n--- AVVIO ORCHESTRAZIONE INTELLIGENTE (LANGGRAPH) ---", flush=True)
    
    initial_state = {
        "task": TASK,
        "analysis_attempts": 0,
        "max_attempts": 3 # Permettiamo 3 tentativi
    }
    
    final_state = app.invoke(initial_state)
    
    print("\n--- ORCHESTRAZIONE COMPLETATA ---", flush=True)
    print(f"\nRISPOSTA FINALE: {final_state['final_answer']}", flush=True)