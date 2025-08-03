import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft, welch
import matplotlib.pyplot as plt
import os

# === 1. MASKELEME ===

def hard_mask(stft_noisy, stft_clean, threshold=1.3):
    ratio = np.abs(stft_clean) / (np.abs(stft_noisy) + 1e-10)
    mask = ratio > threshold
    return stft_noisy * mask

def soft_mask(stft_noisy, stft_clean, threshold=1.2):
    ratio = np.abs(stft_clean) / (np.abs(stft_noisy) + 1e-10)
    gain = np.clip(ratio / threshold, 0, 1)
    return stft_noisy * gain

def bird_mask(stft_noisy, stft_clean, threshold_soft=1.2, threshold_hard=1.4):
    freq_bins, time_bins = stft_noisy.shape
    ratio = np.abs(stft_clean) / (np.abs(stft_noisy) + 1e-10)

    mask = np.ones_like(ratio)
    for f in range(freq_bins):
        freq = f * 22050 / freq_bins
        if 2000 <= freq <= 8000:
            gain = np.clip(ratio[f] / threshold_soft, 0, 1)
        else:
            gain = ratio[f] > threshold_hard
        mask[f] = gain
    return stft_noisy * mask

def protect_speech_band(masked_stft, freqs, speech_band=(300, 3400), protection_factor=0.5):
    protected = masked_stft.copy()
    for i, f in enumerate(freqs):
        if speech_band[0] <= f <= speech_band[1]:
            protected[i, :] = protected[i, :] * (1 + protection_factor)
    return protected

# === 2. GÜrÜLTÜ SINIFLANDIRMA ===
def classify_noise_type(noisy_signal, fs):
    energy = np.mean(noisy_signal ** 2)
    freqs, psd = welch(noisy_signal, fs, nperseg=1024)
    dominant_freq = freqs[np.argmax(psd)]
    spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)

    frame_energy = np.array([
        np.sum(noisy_signal[i:i+512] ** 2)
        for i in range(0, len(noisy_signal) - 512, 512)
    ])
    std_energy = np.std(frame_energy)

    energy_threshold = 3 * np.mean(frame_energy)
    spikes = np.sum(frame_energy > energy_threshold)

    if std_energy > 0.03 and spikes >= 10 and (spectral_centroid > 1500 or (spectral_centroid > 500 and std_energy > 18)):
        return "impulsive"
    elif 1500 < spectral_centroid < 4500 and std_energy > 0.02:
        return "clashing"
    elif spectral_centroid > 6000:
        return "chirp"
    elif spectral_centroid < 600:
        return "continuous_low"
    elif 600 <= spectral_centroid < 1500:
        return "continuous_high"
    else:
        return "continuous_low" if std_energy < 0.015 else "continuous_high"

# === 3. GRAFİK ÇİZİM FONKSİYONU ===
def analyze_and_plot_signal(signal, fs, title="Signal Analysis", output_folder="grafikler", filename_prefix="plot"):
    os.makedirs(output_folder, exist_ok=True)

    freqs, psd = welch(signal, fs, nperseg=1024)
    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs, psd)
    plt.title(f"{title} - Güç Spektral Yoğunluğu")
    plt.xlabel("Frekans (Hz)")
    plt.ylabel("Güç")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{filename_prefix}_psd.png")
    plt.close()

    f, t, Zxx = stft(signal, fs, nperseg=1024)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title(f"{title} - Spektrogram")
    plt.ylabel('Frekans [Hz]')
    plt.xlabel('Zaman [saniye]')
    plt.colorbar(label='Genlik')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{filename_prefix}_spectrogram.png")
    plt.close()

    print(f"\U0001F4CA '{filename_prefix}' için grafikler '{output_folder}/' klasörüne kaydedildi (PSD ve spektrogram).")

# === 4. İŞLEME VE TEMİZLEME ===
def process_and_clean(original_path, noisy_path, output_path):
    fs_o, original = wavfile.read(original_path)
    fs_n, noisy = wavfile.read(noisy_path)

    if original.ndim > 1:
        original = np.mean(original, axis=1)
    if noisy.ndim > 1:
        noisy = np.mean(noisy, axis=1)

    original = original.astype(np.float32)
    noisy = noisy.astype(np.float32)
    original /= (np.max(np.abs(original)) + 1e-10)
    noisy /= (np.max(np.abs(noisy)) + 1e-10)

    analyze_and_plot_signal(noisy, fs_n, title="Gürültülü Sinyal", filename_prefix=os.path.basename(noisy_path).split('.')[0])
    analyze_and_plot_signal(original, fs_o, title="Orijinal Sinyal", filename_prefix=os.path.basename(original_path).split('.')[0])

    f_o, t_o, Zxx_o = stft(original, fs_o, nperseg=1024)
    f_n, t_n, Zxx_n = stft(noisy, fs_n, nperseg=1024)

    noise_type = classify_noise_type(noisy, fs_n)

    if noise_type == "impulsive":
        Zxx_masked = hard_mask(Zxx_n, Zxx_o, threshold=1.3)
    elif noise_type == "continuous_low":
        Zxx_masked = soft_mask(Zxx_n, Zxx_o, threshold=1.1)
    elif noise_type == "continuous_high":
        Zxx_masked = soft_mask(Zxx_n, Zxx_o, threshold=1.2)
    elif noise_type == "chirp":
        Zxx_masked = bird_mask(Zxx_n, Zxx_o, threshold_soft=1.15, threshold_hard=1.35)
    elif noise_type == "clashing":
        raw_masked = hard_mask(Zxx_n, Zxx_o, threshold=1.35)
        Zxx_masked = protect_speech_band(raw_masked, f_n, protection_factor=0.8)
    else:
        Zxx_masked = hard_mask(Zxx_n, Zxx_o, threshold=1.3)

    _, denoised = istft(Zxx_masked, fs_o, nperseg=1024)
    denoised /= (np.max(np.abs(denoised)) + 1e-10)

    analyze_and_plot_signal(denoised, fs_o, title="Temizlenmiş Sinyal", filename_prefix=os.path.basename(output_path).split('.')[0])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wavfile.write(output_path, fs_o, (denoised * 32767).astype(np.int16))
    print(f"✅ {output_path} başarıyla kaydedildi. (Algılanan Gürültü Türü: {noise_type})")

# === 5. TOPLU DOSYA İŞLEME ===
if __name__ == "__main__":
    file_pairs = [
        ("kaynaksesler/0_ORIGINAL.wav",    "kaynaksesler/0_NOISY.wav",    "temizlenen/0_SUPER_CLEANED.wav"),
        ("kaynaksesler/87_ORIGINAL.wav",   "kaynaksesler/87_NOISY.wav",   "temizlenen/87_SUPER_CLEANED.wav"),
        ("kaynaksesler/612_ORIGINAL.wav",  "kaynaksesler/612_NOISY.wav",  "temizlenen/612_SUPER_CLEANED.wav"),
        ("kaynaksesler/616_ORIGINAL.wav",  "kaynaksesler/616_NOISY.wav",  "temizlenen/616_SUPER_CLEANED.wav"),
        ("kaynaksesler/270_ORIGINAL.wav",  "kaynaksesler/270_NOISY.wav",  "temizlenen/270_SUPER_CLEANED.wav"),
        ("kaynaksesler/620_ORIGINAL.wav",  "kaynaksesler/620_NOISY.wav",  "temizlenen/620_SUPER_CLEANED.wav"),
        ("kaynaksesler/573_ORIGINAL.wav",  "kaynaksesler/573_NOISY.wav",  "temizlenen/573_SUPER_CLEANED.wav"),
        ("kaynaksesler/212_ORIGINAL.wav",  "kaynaksesler/212_NOISY.wav",  "temizlenen/212_SUPER_CLEANED.wav"),
    ]

    for original_path, noisy_path, output_path in file_pairs:
        process_and_clean(original_path, noisy_path, output_path)
