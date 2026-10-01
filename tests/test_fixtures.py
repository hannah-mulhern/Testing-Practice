import pytest
from inflammation.models import Patient
import numpy as np

@pytest.fixture()
def patient_1():
    return Patient(id=1, data = [1, 2, 3, 4, 5])

@pytest.fixture()
def patient_2():
    return Patient(id=2, data = [10, 20, 30, 40, 50])

class TestPatient:
    def test_patient_data_mean(self, patient_1):
        assert patient_1.data_mean() == 3.0

    def test_patient_data_max(self, patient_1):
        assert patient_1.data_max() == 5

    def test_patient_data_min(self, patient_1):
        assert patient_1.data_min() == 1

    def test_patient_attributes(self, patient_2):
        assert patient_2.id == 2
        assert patient_2.data == [10, 20, 30, 40, 50]


import numpy.testing as npt
from inflammation.models import Trial 

@pytest.fixture()
def trial_instance():
    return Trial(np.array([[0, 0],[0,0]]), 1)

class TestTrial:
    def test_daily_mean_zeros(self, trial_instance):
        """Test that mean function works for an array of zeros"""
        trial_instance.data = np.array([
            [0, 0],
            [0, 0],
            [0, 0]])
        test_result = np.array([0, 0])

        npt.assert_array_equal(trial_instance.daily_mean(), test_result)
    
    def test_daily_mean_integers(self, trial_instance):
            """Test that mean function works for an array of zeros"""
            trial_instance.data = np.array([
                [1, 2],
                [3, 4],
                [5, 6]])
            test_result = np.array([3, 4])
        
            npt.assert_array_equal(trial_instance.daily_mean(), test_result)

    @pytest.mark.parametrize(
        "test, expected",
        [
            ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]),
            ([[5, 1, 7], [3, 6, 9], [5, 1, 8]], [5, 6, 9]),
            ([[5, 1, -7], [-3, 6, 9], [5, 0, 8]], [5, 6, 9]),
        ])
    def test_daily_max(test, expected):
        """Test max function works for zeroes, positive integers, mix of positive/negative integers."""
        from inflammation.models import daily_max
        npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))



    @pytest.mark.parametrize(
            "test, expected",
            [
                ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]),
                ([[5, 1, 7], [3, 6, 9], [5, 1, 8]], [3, 1, 7]),
                ([[5, 1, -7], [-3, 6, 9], [5, 0, 8]], [-3, 0, -7]),
            ])
    def test_daily_min(test, expected):
        """Test min function works for zeroes, positive integers, mix of positive/negative integers."""
        from inflammation.models import daily_min
        npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))

    @pytest.mark.parametrize(
        "test, expected, expect_raises",
        [
            ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], None),
            ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], None),
            ([[float('nan'), 1, 1], [1, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1], [1, 1, 1]], None),
            ([[1, 2, 3], [4, 5, float('nan')], [7, 8, 9]], [[0.33, 0.67, 1], [0.8, 1, 0], [0.78, 0.89, 1]], None),
            ([[-1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], ValueError),
            ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], None),
            #('testing', None, TypeError),
            #(3, None, TypeError),
        ])
    def test_patient_normalise(test, expected, expect_raises):
        """Test normalisation works for arrays of one and positive integers."""
        from inflammation.models import patient_normalise
        if isinstance(test, list):
            test = np.array(test)
        if expect_raises is not None:
            with pytest.raises(expect_raises):
                npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)
        else:
            npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)



