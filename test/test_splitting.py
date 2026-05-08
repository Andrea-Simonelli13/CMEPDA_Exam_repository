import numpy as np
import json
from unittest.mock import patch, mock_open

from CMEPDA_Exam_repository.train_models import data_splitting 

@patch('matplotlib.pyplot.show')
@patch('numpy.load')
@patch('CMEPDA_Exam_repository.train_models.open', new_callable=mock_open)
def test_data_splitting_integrity(mock_file, mock_np_load, mock_plt):
    """
    Verifica che lo split mantenga il numero totale di campioni 
    e la corrispondenza tra testi e label.
    """
    # 1. Prepariamo 100 articoli fittizi
    total_samples = 100
    fake_json_data = [{"text": f"text {i}", "keywords": ["k1"]} for i in range(total_samples)]
    fake_labels = np.zeros((total_samples, 5)) # 100 articoli, 5 classi

    # Configuriamo i mock per restituire questi dati
    mock_file.return_value.read.return_value = json.dumps(fake_json_data)
    mock_np_load.return_value = fake_labels

    # 2. Eseguiamo lo split
    # (Assumendo che la tua funzione accetti path e restituisca i 6 array classici)
    X_train, X_val, X_test, y_train, y_val, y_test = data_splitting("dummy_path", "dummy_path")

    # 3. ASSERZIONI
    
    # Verifica che il totale sia conservato (100 = 80 + 10 + 10)
    current_total = len(X_train) + len(X_val) + len(X_test)
    assert current_total == total_samples, f"Persi dei dati! {current_total} != {total_samples}"

    # Verifica la corrispondenza X e y per ogni set
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

    # Verifica che y_train abbia il numero corretto di colonne (le 5 classi)
    assert y_train.shape[1] == 5
    assert y_val.shape[1] == 5
    assert y_test.shape[1] == 5
