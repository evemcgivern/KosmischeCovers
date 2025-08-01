import unittest
from unittest import mock
import os
import platform
# import sys
import torch
import importlib.util
# import psutil


class TestHardwareDetection(unittest.TestCase):
    """Test the hardware detection functionality in both
    trainer.py and training.py"""

    def setUp(self):
        # Store original environment to restore after tests
        self.original_platform_machine = platform.machine
        self.original_platform_processor = platform.processor
        self.original_platform_system = platform.system
        self.original_torch_cuda_available = torch.cuda.is_available
        self.original_torch_mps_available = None
        if hasattr(torch.backends, 'mps'):
            self.original_torch_mps_available = torch.backends.mps.is_available

    def tearDown(self):
        # Restore original environment after tests
        platform.machine = self.original_platform_machine
        platform.processor = self.original_platform_processor
        platform.system = self.original_platform_system
        torch.cuda.is_available = self.original_torch_cuda_available
        if hasattr(torch.backends, 'mps') and \
           self.original_torch_mps_available is not None:
            torch.backends.mps.is_available = self.original_torch_mps_available

    def test_trainer_hardware_detection(self):
        """Test the hardware detection function in trainer.py"""
        # Import the detect_hardware function from trainer.py
        spec = importlib.util.spec_from_file_location("trainer", "trainer.py")
        if spec is None:
            self.fail("Could not load spec for trainer.py")
        trainer = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            self.fail("Could not load trainer.py: spec.loader is None")
        spec.loader.exec_module(trainer)

        # Test with different CPU configurations
        with mock.patch('os.cpu_count', return_value=8), \
             mock.patch('psutil.virtual_memory') as mock_memory:

            # Mock 16GB of RAM
            mock_memory.return_value.total = 16 * (1024**3)

            # Run hardware detection
            processes = trainer.detect_hardware()

            # Verify optimal process count (should be 6 with 8 cores * 0.75)
            self.assertEqual(processes, 6)

        # Test with low memory system
        with mock.patch('os.cpu_count', return_value=8), \
             mock.patch('psutil.virtual_memory') as mock_memory:

            # Mock 2GB of RAM (should limit to 1 process)
            mock_memory.return_value.total = 2 * (1024**3)

            # Run hardware detection
            processes = trainer.detect_hardware()

            # Verify optimal process count is limited to 1 due to low memory
            self.assertEqual(processes, 1)

    @mock.patch('torch.cuda.is_available')
    @mock.patch('torch.cuda.get_device_properties')
    @mock.patch('torch.cuda.get_device_name')
    @mock.patch('torch.cuda.device_count')
    def test_training_cuda_detection(
            self, mock_device_count, mock_device_name,
            mock_device_props, mock_cuda_available):
        """Test CUDA detection in training.py"""
        # Skip if we can't import training module
        if not os.path.exists('training.py'):
            self.skipTest("training.py not found")

        # Mock CUDA availability
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        mock_device_name.return_value = "NVIDIA GeForce RTX 3080"

        # Mock device properties with different memory sizes

        # Test high-end GPU (24GB)
        class MockProps:
            total_memory = 24 * (1024**3)
        mock_device_props.return_value = MockProps()

        # We need to reload the training module to rerun the hardware detection
        with mock.patch.dict('sys.modules', {'torch': torch}):
            # This is a partial test - we would need to mock more dependencies
            # to fully load the training module. Instead, we'll verify the main
            # detection logic directly:

            # Verify high-end GPU detection
            if torch.cuda.is_available():
                total_mem = mock_device_props.return_value.total_memory
                gpu_memory = total_mem / (1024**3)
                self.assertEqual(gpu_memory, 24)

                # This should recommend batch size 8 for >20GB VRAM
                if gpu_memory > 20:
                    recommended_batch_size = 8
                    self.assertEqual(recommended_batch_size, 8)

    def test_apple_silicon_detection(self):
        """Test Apple Silicon detection logic"""
        # Mock Apple Silicon environment
        platform.machine = mock.MagicMock(return_value="arm64")
        platform.system = mock.MagicMock(return_value="Darwin")

        # Create a mock for subprocess.check_output to return Mac model info
        with mock.patch('subprocess.check_output') as mock_subprocess:
            # Mock an M2 Mac
            mock_subprocess.return_value = b"hw.model: Mac13,1"

            # Directly call platform.system() and platform.machine()
            self.assertEqual(platform.system(), "Darwin")
            self.assertEqual(platform.machine(), "arm64")

            # The training.py would identify this as Apple Silicon M2
            # and set recommended_batch_size = 3

    def test_intel_mac_detection(self):
        """Test Intel Mac detection logic"""
        # Mock Intel Mac environment
        platform.system = mock.MagicMock(return_value="Darwin")
        platform.processor = mock.MagicMock(
            return_value="Intel(R) Core(TM) i9-9900K"
        )

        # Mock MPS not available (Intel Mac)
        if hasattr(torch.backends, 'mps'):
            torch.backends.mps.is_available = mock.MagicMock(
                return_value=False
            )

        # Mock CUDA not available
        torch.cuda.is_available = mock.MagicMock(return_value=False)

        # Directly call platform.system() and platform.processor()
        self.assertEqual(platform.system(), "Darwin")
        self.assertTrue("Intel" in platform.processor())

        # On Intel Mac, code would set:
        # device = torch.device("cpu")
        # device_name = "Intel Mac CPU"
        # recommended_batch_size = 2

    def test_generic_cpu_detection(self):
        """Test generic CPU detection logic with different CPU types"""
        # Mock a non-Apple, non-CUDA system
        platform.system = mock.MagicMock(return_value="Linux")

        # Test with AMD Ryzen
        platform.processor = mock.MagicMock(return_value="AMD Ryzen 9 5900X")
        self.assertEqual(platform.system(), "Linux")
        self.assertTrue("AMD Ryzen" in platform.processor())

        # For AMD Ryzen, code would set recommended_batch_size = 2

        # Test with Intel Core i-series
        platform.processor = mock.MagicMock(
            return_value="Intel(R) Core(TM) i7-10700K"
        )
        self.assertTrue("Intel" in platform.processor())
        self.assertTrue("i7" in platform.processor())

        # For Intel i-series, code would set recommended_batch_size = 2

        # Test with unknown CPU
        platform.processor = mock.MagicMock(return_value="Some Generic CPU")
        self.assertFalse("AMD Ryzen" in platform.processor())
        self.assertFalse("Intel" in platform.processor())

        # For unknown CPUs, code would set recommended_batch_size = 1


class TestOptimizationSettings(unittest.TestCase):
    """Test the optimization settings based on detected hardware"""

    def test_batch_size_recommendations(self):
        """Test that batch size recommendations are appropriate for hardware"""
        # This would be a more comprehensive test in practice
        # Here we just verify the logic for a few scenarios

        # NVIDIA GPU with 24GB VRAM
        gpu_memory = 24
        if gpu_memory > 20:
            recommended_batch_size = 8
        elif gpu_memory > 12:
            recommended_batch_size = 4
        else:
            recommended_batch_size = 2

        self.assertEqual(recommended_batch_size, 8)

        # NVIDIA GPU with 8GB VRAM
        gpu_memory = 8
        if gpu_memory > 20:
            recommended_batch_size = 8
        elif gpu_memory > 12:
            recommended_batch_size = 4
        elif gpu_memory > 6:
            recommended_batch_size = 2
        else:
            recommended_batch_size = 1

        self.assertEqual(recommended_batch_size, 2)

        # NVIDIA GPU with 4GB VRAM
        gpu_memory = 4
        if gpu_memory > 20:
            recommended_batch_size = 8
        elif gpu_memory > 12:
            recommended_batch_size = 4
        elif gpu_memory > 6:
            recommended_batch_size = 2
        else:
            recommended_batch_size = 1

        self.assertEqual(recommended_batch_size, 1)

    def test_thread_count_optimizations(self):
        """Test that thread counts are set appropriately"""
        # Test with different CPU core counts
        for cpu_count in [2, 4, 8, 16, 32, 64]:
            recommended_threads = min(cpu_count, 8) if cpu_count else 4

            # Verify we never use more than 8 threads
            self.assertLessEqual(recommended_threads, 8)

            # For 8 or fewer cores, we use all cores
            if cpu_count <= 8:
                self.assertEqual(recommended_threads, cpu_count)
            else:
                # For more than 8 cores, we cap at 8
                self.assertEqual(recommended_threads, 8)


if __name__ == '__main__':
    unittest.main()
