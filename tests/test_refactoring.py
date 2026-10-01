import pytest
from inflammation.models import Patient

def test_patient_data_mean():
    patient = Patient(id=1, data = [1, 2, 3, 4, 5])
    assert patient.data_mean() == 3.0

def test_patient_data_max():
    patient = Patient(id=1, data = [1, 2, 3, 4, 5])
    assert patient.data_max() == 5

def test_patient_data_min():
    patient = Patient(id=1, data = [1, 2, 3, 4, 5])
    assert patient.data_min() == 1

def test_patient_attributes():
    patient = Patient(id=2, data = [10, 20, 30, 40, 50])
    assert patient.id == 2
    assert patient.data == [10, 20, 30, 40, 50]