import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('retained_probe', REPO / 'eval/evaluate_frozen_probe.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TinyBackbone(torch.nn.Module):
    num_features = 3
    def forward(self, images):
        return images.mean(dim=(2, 3))


class RetainedProbeTests(unittest.TestCase):
    def fixture(self, root):
        cfg = root / 'model'
        cfg.mkdir()
        (cfg / 'config.json').write_text('{}')
        (cfg / 'model.safetensors').write_bytes(b'fixture-backbone')
        data = root / 'test'
        for name, color in [('Basophil', (240, 0, 0)), ('Monocyte', (0, 0, 240))]:
            (data / name).mkdir(parents=True)
            Image.new('RGB', (19, 23), color).save(data / name / 'sample.png')
        features = np.array([[1.,0.,0.], [0.,0.,1.], [2.,0.,0.], [0.,0.,2.]])
        scaler = StandardScaler().fit(features)
        classifier = LogisticRegression(random_state=0).fit(scaler.transform(features), [0,1,0,1])
        probe = root / 'probe.joblib'
        joblib.dump(dict(scaler=scaler, classifier=classifier, classes=['Basophil','Monocyte'], config={'C': .01, 'weighting':'none'}), probe)
        argv = ['--probe',str(probe),'--probe-sha256',MODULE.digest(probe), '--trust-probe',
                '--model-name','local-dir:' + str(cfg), '--backbone-sha256',MODULE.digest(cfg / 'model.safetensors'),
                '--config-sha256',MODULE.digest(cfg / 'config.json'), '--test-root',str(data),
                '--mode','internal','--device','cpu','--workers','0','--output-dir',str(root / 'out')]
        return argv

    def test_trust_required_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = self.fixture(Path(tmp)); argv.remove('--trust-probe')
            with mock.patch.object(MODULE.joblib, 'load') as load, self.assertRaisesRegex(ValueError, 'trust'):
                MODULE.main(argv)
            load.assert_not_called()

    def test_hash_checked_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = self.fixture(Path(tmp)); argv[argv.index('--probe-sha256')+1] = '0'*64
            with mock.patch.object(MODULE.joblib, 'load') as load, self.assertRaisesRegex(ValueError, 'checksum'):
                MODULE.main(argv)
            load.assert_not_called()

    def test_network_lookup_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = self.fixture(Path(tmp)); argv[argv.index('--model-name')+1] = 'hf-hub:untrusted/model'
            with self.assertRaisesRegex(ValueError, 'locally'):
                MODULE.main(argv)

    def test_internal_inference_does_not_fit_or_select(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); argv=self.fixture(root)
            with mock.patch.object(MODULE.timm,'create_model',return_value=TinyBackbone()) as create, \
                 mock.patch.object(LogisticRegression,'fit',side_effect=AssertionError('refit forbidden')), \
                 mock.patch.object(StandardScaler,'fit',side_effect=AssertionError('refit forbidden')):
                self.assertEqual(MODULE.main(argv),0)
            self.assertTrue(create.call_args.kwargs['pretrained'])
            report=json.loads((root/'out/provenance.json').read_text())
            self.assertFalse(report['fitted_or_selected'])
            self.assertEqual(report['evaluated'],2)
            self.assertEqual(report['class_order'],['Basophil','Monocyte'])

    def test_external_probabilities_average_after_each_view_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); argv=self.fixture(root); argv[argv.index('--mode')+1]='external'
            center=np.array([[1.,0.,0.],[0.,0.,1.]])
            flip=np.array([[2.,0.,0.],[0.,0.,2.]])
            payload=joblib.load(root/'probe.joblib')
            expected=(payload['classifier'].predict_proba(payload['scaler'].transform(center))+
                      payload['classifier'].predict_proba(payload['scaler'].transform(flip)))/2
            with mock.patch.object(MODULE.timm,'create_model',return_value=TinyBackbone()), \
                 mock.patch.object(MODULE.protocol,'embed',side_effect=[center,flip]) as embed:
                MODULE.main(argv)
            self.assertEqual(embed.call_count,2)
            import csv
            with (root/'out/predictions.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            actual=np.array([[float(row['prob_Basophil']),float(row['prob_Monocyte'])] for row in rows])
            np.testing.assert_array_equal(actual,expected)

    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); argv=self.fixture(root); (root/'out').mkdir()
            with self.assertRaisesRegex(ValueError,'new output'):
                MODULE.main(argv)


if __name__ == '__main__':
    unittest.main()
