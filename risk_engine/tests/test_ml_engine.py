from django.test import TestCase
from unittest.mock import MagicMock, patch
import numpy as np
from risk_engine.services.ml_engine import (
    score_patient, _get_artefact_paths, _load_model, FEATURE_NAMES, ML_THRESHOLD
)


def make_consultation(age=70, sex="M", bmi=25.0, albumin_gdl=4.0, haemoglobin_gdl=13.5, comorbidity_count=1, polypharmacy_flag=False, medication_count=3):
    """Create a mock consultation object for testing."""
    c = MagicMock()
    c.patient.age = age
    c.patient.sex = sex
    c.bmi = bmi
    c.albumin_gdl = albumin_gdl
    c.haemoglobin_gdl = haemoglobin_gdl
    c.comorbidity_count = comorbidity_count
    c.polypharmacy_flag = polypharmacy_flag
    c.medication_count = medication_count
    return c


class MLEngineValidationTest(TestCase):
    """Critical validation tests for ML-based nutritional risk assessment."""

    def setUp(self):
        """Reset module-level cache before each test."""
        import risk_engine.services.ml_engine as ml_module
        ml_module._model = None
        ml_module._scaler = None
        ml_module._fnames = None

    @patch('os.path.exists')
    def test_model_artefacts_accessible(self, mock_exists):
        """Verify model files are correctly located."""
        mock_exists.return_value = True
        paths = _get_artefact_paths()

        required_files = ['model.pkl', 'scaler.pkl', 'feature_names.json']
        for file in required_files:
            self.assertTrue(any(file in path for path in paths.values()))

    @patch('os.path.exists')
    @patch('json.load')
    @patch('joblib.load')
    def test_model_loading_validation(self, mock_joblib_load, mock_json_load, mock_exists):
        """Ensure model loads with correct feature alignment."""
        mock_exists.return_value = True
        mock_json_load.return_value = FEATURE_NAMES

        mock_model = MagicMock()
        mock_scaler = MagicMock()
        mock_joblib_load.side_effect = [mock_model, mock_scaler]

        model, scaler, fnames = _load_model()

        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)
        self.assertEqual(fnames, FEATURE_NAMES)

    @patch('os.path.exists')
    def test_missing_model_error_handling(self, mock_exists):
        """Verify system fails safely when model files are missing."""
        mock_exists.return_value = False

        with self.assertRaises(RuntimeError) as cm:
            _load_model()

        self.assertIn('model not found', str(cm.exception).lower())


class ScoringFunctionalityTest(TestCase):
    """Core functionality tests for patient risk scoring."""

    def setUp(self):
        """Set up mock ML components."""
        import risk_engine.services.ml_engine as ml_module
        ml_module._model = None
        ml_module._scaler = None
        ml_module._fnames = None

        self.mock_model = MagicMock()
        self.mock_scaler = MagicMock()

        # Mock high-risk prediction (70% probability)
        self.mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        self.mock_model.coef_ = np.array([[0.5, -0.3, 0.8, -0.6, 0.2, 0.4, -0.1]])
        self.mock_scaler.transform.return_value = np.array([[1.2, 0.8, -0.5, 1.1, 0.3, 0.9, -0.2]])

    @patch('risk_engine.services.ml_engine._load_model')
    def test_risk_scoring_workflow(self, mock_load_model):
        """Test complete risk scoring process."""
        mock_load_model.return_value = (self.mock_model, self.mock_scaler, FEATURE_NAMES)

        consultation = make_consultation()
        result = score_patient(consultation)

        # Verify core outputs
        self.assertIn('probability', result)
        self.assertIn('risk_flag', result)
        self.assertIn('top_factors', result)

        # Verify risk assessment
        self.assertEqual(result['probability'], 0.7)
        self.assertTrue(result['risk_flag'])

    @patch('risk_engine.services.ml_engine._load_model')
    def test_missing_data_handling(self, mock_load_model):
        """Test fallback values for missing patient data."""
        mock_load_model.return_value = (self.mock_model, self.mock_scaler, FEATURE_NAMES)

        # Test with missing BMI and albumin
        consultation = make_consultation(bmi=None, albumin_gdl=None)
        result = score_patient(consultation)

        # Should still produce valid results
        self.assertIsInstance(result['probability'], float)
        self.assertIsInstance(result['risk_flag'], bool)

    @patch('risk_engine.services.ml_engine._load_model')
    def test_feature_vector_assembly(self, mock_load_model):
        """Verify correct feature extraction and ordering."""
        mock_load_model.return_value = (self.mock_model, self.mock_scaler, FEATURE_NAMES)

        consultation = make_consultation(
            age=75, sex="F", bmi=22.5, albumin_gdl=3.8,
            haemoglobin_gdl=12.5, comorbidity_count=2, polypharmacy_flag=True
        )

        score_patient(consultation)

        # Verify feature vector passed to model
        call_args = self.mock_scaler.transform.call_args[0][0]
        expected = np.array([[75, 0, 22.5, 3.8, 12.5, 2, 1]])  # sex_encoded=0 for Female
        np.testing.assert_array_equal(call_args, expected)

    @patch('risk_engine.services.ml_engine._load_model')
    def test_risk_threshold_configuration(self, mock_load_model):
        """Test configurable risk threshold."""
        mock_load_model.return_value = (self.mock_model, self.mock_scaler, FEATURE_NAMES)

        consultation = make_consultation()

        # Test with custom threshold
        result = score_patient(consultation, threshold=0.8)
        self.assertFalse(result['risk_flag'])  # 0.7 < 0.8

    @patch('risk_engine.services.ml_engine._load_model')
    def test_top_factors_analysis(self, mock_load_model):
        """Verify factor contribution analysis."""
        mock_load_model.return_value = (self.mock_model, self.mock_scaler, FEATURE_NAMES)

        consultation = make_consultation()
        result = score_patient(consultation)

        factors = result['top_factors']
        self.assertEqual(len(factors), 4)

        # Verify factor structure
        for factor in factors:
            self.assertIn('feature', factor)
            self.assertIn('contribution', factor)
            self.assertIn('direction', factor)

        # Verify sorting by contribution magnitude
        contributions = [abs(f['contribution']) for f in factors]
        self.assertEqual(contributions, sorted(contributions, reverse=True))


class SystemConfigurationTest(TestCase):
    """Configuration and constants validation."""

    def test_feature_specification(self):
        """Verify feature set matches training configuration."""
        expected_features = [
            "age", "sex_encoded", "bmi", "albumin_gdl",
            "haemoglobin_gdl", "comorbidity_count", "polypharmacy_5plus"
        ]
        self.assertEqual(FEATURE_NAMES, expected_features)

    def test_default_risk_threshold(self):
        """Verify default risk classification threshold."""
        self.assertEqual(ML_THRESHOLD, 0.45)
        self.assertIsInstance(ML_THRESHOLD, float)