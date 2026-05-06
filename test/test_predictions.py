import pytest
import numpy as np
from unittest.mock import patch, mock_open#, MagicMock

from CMEPDA_Exam_repository.new_model_predictions import (
    trained_CNN_model_prediction,
    trained_Dense_model_prediction,
    trained_LSTM_model_prediction
    )

#@patch('CMEPDA_Exam_repository.new_model_predictions.plt.show') # Evitiamo i grafici
#@patch('CMEPDA_Exam_repository.new_model_predictions.tensorflow.keras.models.load_weights')
#@patch('CMEPDA_Exam_repository.new_model_predictions.pickle.load') # Mock vocabolario
#@patch('CMEPDA_Exam_repository.new_model_predictions.np.load')     # Mock classi e labels
@pytest.mark.parametrize("prediction_function", [
    trained_Dense_model_prediction,
    trained_CNN_model_prediction,
    trained_LSTM_model_prediction
])
@patch('matplotlib.pyplot.show')
@patch('tensorflow.keras.Sequential.load_weights')
@patch('pickle.load')
@patch('numpy.load')
@patch('CMEPDA_Exam_repository.new_model_predictions.data_splitting_pred')
@patch('CMEPDA_Exam_repository.new_model_predictions.open', new_callable=mock_open)

@pytest.mark.filterwarnings("ignore:Precision is ill-defined")
@pytest.mark.filterwarnings("ignore:Recall is ill-defined")
@pytest.mark.filterwarnings("ignore:F-score is ill-defined")

def test_prediction_edge_cases(mock_file, mock_split, mock_np_load, mock_pickle, mock_weights, mock_plt, prediction_function):
    """
    Testa la robustezza della predizione con input estremi o vuoti.
    """
    
    # 1. CONFIGURAZIONE MOCK (Scenario base: 2 classi, vocabolario di 25000)
    mock_pickle.return_value = [f"word{i}" for i in range(24998)]#["word"] * 25000
    mock_np_load.side_effect = [
        #np.array([[0, 1]]), # y_test (matrice 1x2)
        np.array(["math", "physics"]) # classi (2 classi)
    ]
    
    # 2. CASO LIMITE: TESTO VUOTO O CORTISSIMO
    # Simuliamo che data_splitting_pred restituisca un abstract vuoto ""
    # o un abstract irrilevante per la fisica
    mock_split.return_value = (
        None, None, [""] * 30, # X_train, X_val, X_test (abstract vuoto)
        np.array([[0, 0]]), # y_train (non usato nel corpo ma richiesto per la shape)
        np.array([[0, 0]]), # y_val
        np.array([[0, 0]] * 30), # y_test
        [], [], [""] * 30        # orig_test
    )

    # 3. ESECUZIONE
    # Invece di far girare tutto, vogliamo verificare che la funzione gestisca
    # il processo senza sollevare errori di dimensione (ShapeMismatch)
    try:
        # Patchiamo il modello per restituire probabilità molto basse (vicine a 0)
        # dato che l'input è vuoto
        with patch('tensorflow.keras.Sequential.predict') as mock_predict:
            mock_predict.return_value = np.array([[0.01, 0.02]] * 30) # Basse confidenze
            
            #trained_Dense_model_prediction()
            prediction_function()
            
            # 4. VERIFICHE LOGICHE
            # Se la soglia è 0.5, non dovrebbero esserci keywords predette
            # Possiamo verificare che la funzione abbia stampato "Keywords predette: ()" 
            # o semplicemente che non sia crashata.
            assert mock_predict.called
            
    except Exception as e:
        pytest.fail(f"La predizione è crashata con input vuoto: {e}")

@pytest.mark.parametrize("fake_probabilities, threshold, expected_count", [
    (np.array([[0.9, 0.8]]), 0.5, 2), # Entrambe sopra soglia
    (np.array([[0.4, 0.1]]), 0.5, 0), # Nessuna sopra soglia
    (np.array([[0.51, 0.1]]), 0.5, 1), # Solo una sopra soglia
])
def test_threshold_logic(fake_probabilities, threshold, expected_count):
    """
    Verifica che il filtraggio per soglia restituisca il numero corretto di label.
    """
    # Questa logica testa la parte: (prediction > soglia).astype(int)
    binary_output = (fake_probabilities > threshold).astype(int)
    assert np.sum(binary_output) == expected_count
