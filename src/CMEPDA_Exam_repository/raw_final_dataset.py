'''Questo programma invia una richiesta HTTP all' API di HEP-INSPIRE e scarica articoli
del catalogo hep-ph di arXiv. Per rispettare i limiti dell' API si suddivide la richiesta
in anni di pubblicazione. Per ogni articolo si filtrano i metadati per ottenere il titolo
di arXiv e le keywords di HEP-INSPIRE che vengono salvate in un dizionario. I dizionari di ogni
articolo sono inseriti in una lista e la lista è salvata in un file .json. 
'''
import os
import json
import time
import requests
#import shutil

def download_hep_ph_batches(batch_size, max_papers): #, save_folder='data/raw/hep_ph_json'
    '''Funzione che scarica e filtra gli articoli di HEP-INSPIRE.
    Argomenti
    ---------
    batch_size : numero
                 Questo è il numero di articoli per richiesta. Deve essere al massimo 1000,
                 ma meglio sotto i 700.
    max_papers : numero
                 Questo è il numero di articoli scaricati da HEP-INPIRE per ogni anno.
                 L'API di HEP-INSPIRE impone un massimo di 1000. Se max_papers non è un multiplo
                 di batch_size, il numero totale di articoli sarà il primo multiplo di batch_size
                 più grande di max_papers. Si consiglia quindi di impostare un valore massimo intorno
                 ai 9000.
    '''

    save_folder = f"data/raw/raw_final_dataset.json"
    #controlla se esistono le cartelle e le cancella compresi i file al loro interno
    #if os.path.exists(save_folder):
        #shutil.rmtree(save_folder)
    # Cartella dove salvare i batch JSON
    #os.makedirs(save_folder, exist_ok=True)

    # cancella i file JSON esistenti
    #for file in os.listdir(save_folder):
        #file_path = os.path.join(save_folder, file)
        #if os.path.isfile(file_path):
            #os.remove(file_path)

    #print("Cartella pulita. Inizio download...")

    #batch_size = 100   # quanti articoli per batch
    #max_papers = 2000  # massimo articoli da scaricare
    dataset=[]
    for year in range(1991, 2025):
        print(f"Scaricando articoli anno {year}")
        downloaded = 0
        page = 1

        while downloaded < max_papers:
            print(f"Scaricando batch {page}...")

            url = "https://inspirehep.net/api/literature/"
            params = {
                "q": f"arxiv_eprints.categories:hep-ph AND date:{year}", #value
                "size": batch_size,
                "page": page,
                "format": "json" #"json-expanded"
            }

            #response = requests.get(url, params=params)
            #Si fa un try-box per la richiesta HTTP. Viene inviata la richiesta, se non si riceve una risposta
            #dall'API in 2 minuti viene sollevata un'exception. Nell'exception si aspetta 30s e si invia una seconde
            #richiesta. Se alla terza richiesta non si è ricevuto risposta, si passa alla pagina successiva.
            success = False
            for tentativi in range(3):
                try:
                    response = requests.get(url, params=params, timeout=120)
                    response.raise_for_status()
                    success = True
                    break 
                except requests.exceptions.RequestException as e:
                    attesa = 30 * (tentativi + 1)
                    print(f"Errore al tentativo {tentativi + 1}: {e}")
                    print(f"Riprovo tra {attesa} secondi...")
                    time.sleep(attesa)
                    #print(f"Errore API: {e}")
                    #time.sleep(30)
                    #continue
                    #break

            if not success:
                print(f"ATTENZIONE: Impossibile scaricare batch {page} dopo 3 tentativi. Salto al batch successivo.")
                page += 1 # Salta questa pagina per non restare bloccato all'infinito
                continue 
            #if response.status_code != 200:
                #print(f"Errore API: {response.status_code}")
                #break

            data = response.json()
            papers = data.get("hits", {}).get("hits", [])

            if not papers:
                print("Nessun altro paper trovato, fine download.")
                break

            # Salva batch JSON
            #batch_file = f"{save_folder}/batch_{page}.json"
            #with open(batch_file, "w", encoding="utf-8") as f:
                #json.dump(data, f, ensure_ascii=False, indent=2)

            for paper in papers:
                metadata = paper.get("metadata", {}) #prendo l'articolo

                #title = metadata.get("titles", [{}])[0].get("title", "")
                #titles può contenere più di un titolo.
                #Quindi si prende il primo titolo con "source" == "arXiv"
                arxiv_title = ""
                titles = metadata.get("titles", [{}])
                for title in titles:
                    if title.get("source", "") == "arXiv":
                        arxiv_title = title.get("title", "")
                        break
                #abstract = metadata.get("abstracts", [{}])[0].get("value", "")
                #abstracts può contenere più di un titolo.
                #Quindi si prende il primo abstract con "source" == "arXiv"
                arxiv_abstract = ""
                abstracts = metadata.get("abstracts", [{}])
                for abstract in abstracts:
                    if abstract.get("source", "") == "arXiv":
                        arxiv_abstract = abstract.get("value", "")
                        break
                keywords = [
                    k.get("value")
                    for k in metadata.get("keywords", [])
                    if k.get("schema") == "INSPIRE" and k.get("value")
                ]

                if not arxiv_abstract or not keywords or not arxiv_title:
                    continue

                text = arxiv_title + " " + arxiv_abstract
                dataset.append({
                    "text": text,
                    "keywords": keywords#", ".join(keywords)
                })

            downloaded += len(papers)
            print(f"Batch {page} scaricato ({len(papers)} articoli)")

            page += 1
            time.sleep(7)  # pausa per rispettare limiti API

    print(f"numero di papers {len(dataset)}")
    with open(save_folder, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Download completato, tutti i batch salvati in {save_folder}")

if __name__ == "__main__":
    
    download_hep_ph_batches(
        batch_size=50, #100
        max_papers=5000, #9000
    )