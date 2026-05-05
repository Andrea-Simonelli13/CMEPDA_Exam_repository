from unittest.mock import patch, MagicMock, mock_open
import requests

from CMEPDA_Exam_repository.raw_final_dataset import download_hep_ph_batches

@patch('CMEPDA_Exam_repository.raw_final_dataset.requests.get')
@patch('CMEPDA_Exam_repository.raw_final_dataset.time.sleep', return_value=None)  # Salta l'attesa di 30s tra i tentativi
@patch('builtins.open', new_callable=mock_open) # Non scrivere file reali
def test_download_retry_mechanism(mock_sleep, mock_get):
    """
    Testa che la funzione riprovi fino a 3 volte in caso di errore
    e che prosegua se al terzo tentativo ha successo.
    """

    # 1. Definiamo i comportamenti dei tentativi:
    # Tentativo 1: Errore di connessione
    # Tentativo 2: Errore di timeout
    # Tentativo 3: Risposta corretta (200 OK) ma senza paper per fermare il loop
    mock_get.side_effect = [
        requests.exceptions.ConnectionError("Errore 1"),
        requests.exceptions.Timeout("Errore 2"),
        MagicMock(
            status_code=200,
            json=lambda: {"hits": {"hits": []}} # Lista vuota per uscire dal while
        )
    ]

    # 2. Eseguiamo la funzione con parametri minimi per velocità
    # Usiamo un solo anno (se possibile) o limitiamo max_papers
    # Nota: se la tua funzione ha il range(1991, 2025) fisso,
    # il test proverà a fare il download per tutti gli anni.
    # Mockiamo il range o la durata per il test.
    with patch('CMEPDA_Exam_repository.raw_final_dataset.range') as mock_range: # Testa solo l'anno 2024
        mock_range.side_effect = [[2024], range(3), range(3), range(3)]
        download_hep_ph_batches(batch_size=10, max_papers=10)

    # 3. Verifiche (Assertions)

    # Verifichiamo che requests.get sia stato chiamato 3 volte prima di avere successo
    assert mock_get.call_count == 3

    # Verifichiamo che time.sleep sia stato chiamato per le attese (30s e 60s)
    # La tua funzione fa: 30 * (tentativi + 1)
    assert mock_sleep.call_args_list[0][0][0] == 30
    assert mock_sleep.call_args_list[1][0][0] == 60
