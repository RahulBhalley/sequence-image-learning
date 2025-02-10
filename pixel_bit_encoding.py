import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelEncoder:
    @staticmethod
    def encode_rgb(x: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image pixels (values 0-255) into 24-bit binary representation.
        Input shape: [..., 3] (RGB channels)
        Output shape: [..., 24] (8 bits per channel)
        """
        # Ensure input is in correct range
        if x.dtype != torch.uint8:
            x = (x * 255).clamp(0, 255).to(torch.uint8)
        
        # Prepare binary representation
        bits = torch.zeros((*x.shape[:-1], 24), dtype=torch.float32, device=x.device)
        
        # For each color channel
        for i in range(3):
            channel = x[..., i]
            for bit in range(8):
                bits[..., i*8 + bit] = (channel & (1 << bit)) > 0
        
        return bits

    @staticmethod
    def decode_rgb(bits: torch.Tensor) -> torch.Tensor:
        """
        Decode 24-bit binary representation back to RGB image pixels.
        Input shape: [..., 24] (8 bits per channel)
        Output shape: [..., 3] (RGB channels)
        """
        # Initialize output tensor
        rgb = torch.zeros((*bits.shape[:-1], 3), dtype=torch.float32, device=bits.device)
        
        # For each color channel
        for i in range(3):
            channel_bits = bits[..., i*8:(i+1)*8]
            values = torch.zeros_like(channel_bits[..., 0])
            for bit in range(8):
                values += channel_bits[..., bit] * (1 << bit)
            rgb[..., i] = values
        
        return rgb / 255.0

    @staticmethod
    def encode_grayscale(x: torch.Tensor) -> torch.Tensor:
        """
        Encode grayscale image pixels (values 0-255) into 8-bit binary representation.
        Input shape: [..., 1] (grayscale channel)
        Output shape: [..., 8] (8 bits)
        """
        # Ensure input is in correct range
        if x.dtype != torch.uint8:
            x = (x * 255).clamp(0, 255).to(torch.uint8)
        
        # Prepare binary representation
        bits = torch.zeros((*x.shape[:-1], 8), dtype=torch.float32, device=x.device)
        
        # Convert to bits
        x = x.squeeze(-1)  # Remove channel dimension
        for bit in range(8):
            bits[..., bit] = (x & (1 << bit)) > 0
        
        return bits

    @staticmethod
    def decode_grayscale(bits: torch.Tensor) -> torch.Tensor:
        """
        Decode 8-bit binary representation back to grayscale image pixels.
        Input shape: [..., 8] (8 bits)
        Output shape: [..., 1] (grayscale channel)
        """
        # Initialize values
        values = torch.zeros((*bits.shape[:-1], 1), dtype=torch.float32, device=bits.device)
        
        # Convert from bits
        for bit in range(8):
            values[..., 0] += bits[..., bit] * (1 << bit)
        
        return values / 255.0
    
    # Test comparision between encoding and decoding
    @staticmethod
    def test_encoding_decoding():
        # Test RGB encoding/decoding
        rgb_image = torch.randint(0, 256, (2, 3, 3), dtype=torch.uint8)  # 2 pixels, RGB
        encoded_rgb = PixelEncoder.encode_rgb(rgb_image)    
        decoded_rgb = PixelEncoder.decode_rgb(encoded_rgb)
        decoded_rgb_uint8 = (decoded_rgb * 255).to(torch.uint8)  # Convert back to uint8
        assert torch.allclose(rgb_image, decoded_rgb_uint8)  # Compare with converted tensor

# Example usage
if __name__ == "__main__":
    # Test RGB encoding/decoding
    rgb_image = torch.randint(0, 256, (2, 3, 3), dtype=torch.uint8)  # 2 pixels, RGB
    encoded_rgb = PixelEncoder.encode_rgb(rgb_image)
    decoded_rgb = PixelEncoder.decode_rgb(encoded_rgb)
    print("RGB image:", rgb_image)
    # print("Encoded RGB image:", encoded_rgb)
    # print("Decoded RGB image:", decoded_rgb)
    # print("RGB image shape:", rgb_image.shape)
    print("Encoded RGB image shape:", encoded_rgb.shape)
    print("Decoded RGB image shape:", decoded_rgb.shape)
    
    # Test grayscale encoding/decoding
    gray_image = torch.randint(0, 256, (2, 1), dtype=torch.uint8)  # 2 pixels, grayscale
    encoded_gray = PixelEncoder.encode_grayscale(gray_image)
    decoded_gray = PixelEncoder.decode_grayscale(encoded_gray)
    PixelEncoder.test_encoding_decoding()
    
    print("RGB encoding shape:", encoded_rgb.shape)  # Should be (2, 24)
    print("Grayscale encoding shape:", encoded_gray.shape)  # Should be (2, 8)