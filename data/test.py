from efft import Stimulus, Stimuli, eFFT
import matplotlib.pyplot as plt
import numpy as np


# 1. init eFFT
efft = eFFT(128)      # frame size = 512
efft.initialize()

# 2. create 10k random events
stimuli = Stimuli()
for _ in range(100000):
    x = np.random.randint(0, 128)
    y = np.random.randint(0, 128)
    p = np.random.rand() > 0.5
    stimuli.append(Stimulus(x, y, p))

# feed events
efft.update(stimuli)

# 3. get FFT and make it a torch tensor
fft_result = efft.get_fft()  # shape ~ (512, 512), complex
# for magnitude spectrum
magnitude = np.abs(fft_result)

# 4. visualize FFT magnitude
plt.imshow(np.log1p(magnitude), cmap='gray')
plt.title('FFT Magnitude Spectrum after moving dot')
plt.colorbar()
plt.savefig('fft_magnitude_spectrum_moving_dot.png')
plt.close()

# 5. inverse FFT (use 2D, not 1D)
ifft_result = np.fft.ifft2(fft_result)
print(ifft_result.shape)

# 6. visualize inverse FFT (real part)
ifft_img = ifft_result.real
plt.imshow(ifft_img, cmap='gray')
plt.title('Inverse FFT Result')
plt.colorbar()
plt.savefig('ifft_result.png')
plt.close()
