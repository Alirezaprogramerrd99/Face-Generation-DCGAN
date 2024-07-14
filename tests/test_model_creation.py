import unittest
from unittest.mock import patch
import torch
# from main_model.DCGAN_model import *
import os, sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

from main_model.DCGAN_model import *



# Unit tests
class TestMultiGPUHandling(unittest.TestCase):

    @patch('torch.cuda.device_count', return_value=4)
    def test_single_gpu(self, generator_discriminator=True):
        print("\nrunning the test number 1")
        device_test = torch.device('cuda')
        ngpu_test = 1
        net = create_generator_discriminator(ngpu=ngpu_test, device=device_test, 
                                             generator_discriminator=generator_discriminator, 
                                             testing=True)
        # self.assertIsInstance(net, Generator) if generator_discriminator else self.assertIsInstance(net, Discriminator)
        self.assertIsInstance(net, Generator)

    @patch('torch.cuda.device_count', return_value=3)
    def test_multiple_gpus(self, generator_discriminator=True):
        print("\nrunning the test number 2")
        device_test = torch.device('cuda')
        ngpu_test = 3
        net = create_generator_discriminator(ngpu=ngpu_test, device=device_test, 
                                             generator_discriminator=generator_discriminator, 
                                             testing=True)
        self.assertIsInstance(net, nn.DataParallel)

    @patch('torch.cuda.device_count', return_value=1)
    def test_invalid_ngpu(self, generator_discriminator=True):
        print("\nrunning the test number 3")
        device_test = torch.device('cuda')
        ngpu_test = 3
        with self.assertRaises(ValueError):
            # print('Ok, the ValueError raised!')
            create_generator_discriminator(ngpu=ngpu_test, device=device_test, 
                                           generator_discriminator=generator_discriminator, 
                                           testing=True)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)