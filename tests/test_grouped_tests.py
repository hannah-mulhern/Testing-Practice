import pytest

from inflammation.models import Patient

class TestPatient:
    def setup_class(self):
        self.patient1 = Patient(id=1, data = [1, 2, 3, 4, 5])
        self.patient2 = Patient(id=2, data = [10, 20, 30, 40, 50])

    def test_patient_data_mean(self):
        assert self.patient1.data_mean() == 3.0

    def test_patient_data_max(self):
        assert self.patient1.data_max() == 5

    def test_patient_data_min(self):
        assert self.patient1.data_min() == 1

    def test_patient_attributes(self):
        assert self.patient2.id == 2
        assert self.patient2.data == [10, 20, 30, 40, 50]