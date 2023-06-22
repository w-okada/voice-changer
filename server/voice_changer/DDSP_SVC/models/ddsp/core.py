import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import numpy as np

def MaskedAvgPool1d(x, kernel_size):
    x = x.unsqueeze(1)
    x = F.pad(x, ((kernel_size - 1) // 2, kernel_size // 2), mode="reflect")
    mask = ~torch.isnan(x)
    masked_x = torch.where(mask, x, torch.zeros_like(x))
    ones_kernel = torch.ones(x.size(1), 1, kernel_size, device=x.device)

    # Perform sum pooling
    sum_pooled = F.conv1d(
        masked_x,
        ones_kernel,
        stride=1,
        padding=0,
        groups=x.size(1),
    )

    # Count the non-masked (valid) elements in each pooling window
    valid_count = F.conv1d(
        mask.float(),
        ones_kernel,
        stride=1,
        padding=0,
        groups=x.size(1),
    )
    valid_count = valid_count.clamp(min=1)  # Avoid division by zero

    # Perform masked average pooling
    avg_pooled = sum_pooled / valid_count

    return avg_pooled.squeeze(1)

def MedianPool1d(x, kernel_size):
    x = x.unsqueeze(1)
    x = F.pad(x, ((kernel_size - 1) // 2, kernel_size // 2), mode="reflect")
    x = x.squeeze(1)
    x = x.unfold(1, kernel_size, 1)
    x, _ = torch.sort(x, dim=-1)
    return x[:, :, (kernel_size - 1) // 2]
    
def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True):
  """Calculate final size for efficient FFT.
  Args:
    frame_size: Size of the audio frame.
    ir_size: Size of the convolving impulse response.
    power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
      numbers. TPU requires power of 2, while GPU is more flexible.
  Returns:
    fft_size: Size for efficient FFT.
  """
  convolved_frame_size = ir_size + frame_size - 1
  if power_of_2:
    # Next power of 2.
    fft_size = int(2**np.ceil(np.log2(convolved_frame_size)))
  else:
    fft_size = convolved_frame_size
  return fft_size


def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(torch.cat((signal,signal[:,:,-1:]),2), size=signal.shape[-1] * factor + 1, mode='linear', align_corners=True)
    signal = signal[:,:,:-1]
    return signal.permute(0, 2, 1)


def remove_above_fmax(amplitudes, pitch, fmax, level_start=1):
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(level_start, n_harm + level_start).to(pitch)
    aa = (pitches < fmax).float() + 1e-7
    return amplitudes * aa


def crop_and_compensate_delay(audio, audio_size, ir_size,
                              padding = 'same',
                              delay_compensation = -1):
  """Crop audio output from convolution to compensate for group delay.
  Args:
    audio: Audio after convolution. Tensor of shape [batch, time_steps].
    audio_size: Initial size of the audio before convolution.
    ir_size: Size of the convolving impulse response.
    padding: Either 'valid' or 'same'. For 'same' the final output to be the
      same size as the input audio (audio_timesteps). For 'valid' the audio is
      extended to include the tail of the impulse response (audio_timesteps +
      ir_timesteps - 1).
    delay_compensation: Samples to crop from start of output audio to compensate
      for group delay of the impulse response. If delay_compensation < 0 it
      defaults to automatically calculating a constant group delay of the
      windowed linear phase filter from frequency_impulse_response().
  Returns:
    Tensor of cropped and shifted audio.
  Raises:
    ValueError: If padding is not either 'valid' or 'same'.
  """
  # Crop the output.
  if padding == 'valid':
    crop_size = ir_size + audio_size - 1
  elif padding == 'same':
    crop_size = audio_size
  else:
    raise ValueError('Padding must be \'valid\' or \'same\', instead '
                     'of {}.'.format(padding))

  # Compensate for the group delay of the filter by trimming the front.
  # For an impulse response produced by frequency_impulse_response(),
  # the group delay is constant because the filter is linear phase.
  total_size = int(audio.shape[-1])
  crop = total_size - crop_size
  start = (ir_size // 2 if delay_compensation < 0 else delay_compensation)
  end = crop - start
  return audio[:, start:-end]


def fft_convolve(audio,
                 impulse_response): # B, n_frames, 2*(n_mags-1)
    """Filter audio with frames of time-varying impulse responses.
    Time-varying filter. Given audio [batch, n_samples], and a series of impulse
    responses [batch, n_frames, n_impulse_response], splits the audio into frames,
    applies filters, and then overlap-and-adds audio back together.
    Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute
    convolution for large impulse response sizes.
    Args:
        audio: Input audio. Tensor of shape [batch, audio_timesteps].
        impulse_response: Finite impulse response to convolve. Can either be a 2-D
        Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
        ir_frames, ir_size]. A 2-D tensor will apply a single linear
        time-invariant filter to the audio. A 3-D Tensor will apply a linear
        time-varying filter. Automatically chops the audio into equally shaped
        blocks to match ir_frames.
    Returns:
        audio_out: Convolved audio. Tensor of shape
            [batch, audio_timesteps].
    """
    # Add a frame dimension to impulse response if it doesn't have one.
    ir_shape = impulse_response.size() 
    if len(ir_shape) == 2:
        impulse_response = impulse_response.unsqueeze(1)
        ir_shape = impulse_response.size()

    # Get shapes of audio and impulse response.
    batch_size_ir, n_ir_frames, ir_size = ir_shape
    batch_size, audio_size = audio.size() # B, T

    # Validate that batch sizes match.
    if batch_size != batch_size_ir:
        raise ValueError('Batch size of audio ({}) and impulse response ({}) must '
                        'be the same.'.format(batch_size, batch_size_ir))

    # Cut audio into 50% overlapped frames (center padding).
    hop_size = int(audio_size / n_ir_frames)
    frame_size = 2 * hop_size    
    audio_frames = F.pad(audio, (hop_size, hop_size)).unfold(1, frame_size, hop_size)
    
    # Apply Bartlett (triangular) window
    window = torch.bartlett_window(frame_size).to(audio_frames)
    audio_frames = audio_frames * window
    
    # Pad and FFT the audio and impulse responses.
    fft_size = get_fft_size(frame_size, ir_size, power_of_2=False)
    audio_fft = torch.fft.rfft(audio_frames, fft_size)
    ir_fft = torch.fft.rfft(torch.cat((impulse_response,impulse_response[:,-1:,:]),1), fft_size)
    
    # Multiply the FFTs (same as convolution in time).
    audio_ir_fft = torch.multiply(audio_fft, ir_fft)

    # Take the IFFT to resynthesize audio.
    audio_frames_out = torch.fft.irfft(audio_ir_fft, fft_size)
    
    # Overlap Add
    batch_size, n_audio_frames, frame_size = audio_frames_out.size() # # B, n_frames+1, 2*(hop_size+n_mags-1)-1
    fold = torch.nn.Fold(output_size=(1, (n_audio_frames - 1) * hop_size + frame_size),kernel_size=(1, frame_size),stride=(1, hop_size))
    output_signal = fold(audio_frames_out.transpose(1, 2)).squeeze(1).squeeze(1)
    
    # Crop and shift the output audio.
    output_signal = crop_and_compensate_delay(output_signal[:,hop_size:], audio_size, ir_size)
    return output_signal
    

def apply_window_to_impulse_response(impulse_response, # B, n_frames, 2*(n_mag-1)
                                     window_size: int = 0,
                                     causal: bool = False):
    """Apply a window to an impulse response and put in causal form.
    Args:
        impulse_response: A series of impulse responses frames to window, of shape
        [batch, n_frames, ir_size]. ---------> ir_size means size of filter_bank ??????
        
        window_size: Size of the window to apply in the time domain. If window_size
        is less than 1, it defaults to the impulse_response size.
        causal: Impulse response input is in causal form (peak in the middle).
    Returns:
        impulse_response: Windowed impulse response in causal form, with last
        dimension cropped to window_size if window_size is greater than 0 and less
        than ir_size.
    """
    
    # If IR is in causal form, put it in zero-phase form.
    if causal:
        impulse_response = torch.fftshift(impulse_response, axes=-1)
    
    # Get a window for better time/frequency resolution than rectangular.
    # Window defaults to IR size, cannot be bigger.
    ir_size = int(impulse_response.size(-1))
    if (window_size <= 0) or (window_size > ir_size):
        window_size = ir_size
    window = nn.Parameter(torch.hann_window(window_size), requires_grad = False).to(impulse_response)
    
    # Zero pad the window and put in in zero-phase form.
    padding = ir_size - window_size
    if padding > 0:
        half_idx = (window_size + 1) // 2
        window = torch.cat([window[half_idx:],
                            torch.zeros([padding]),
                            window[:half_idx]], axis=0)
    else:
        window = window.roll(window.size(-1)//2, -1)
        
    # Apply the window, to get new IR (both in zero-phase form).
    window = window.unsqueeze(0)
    impulse_response = impulse_response * window
    
    # Put IR in causal form and trim zero padding.
    if padding > 0:
        first_half_start = (ir_size - (half_idx - 1)) + 1
        second_half_end = half_idx + 1
        impulse_response = torch.cat([impulse_response[..., first_half_start:],
                                    impulse_response[..., :second_half_end]],
                                    dim=-1)
    else:
        impulse_response = impulse_response.roll(impulse_response.size(-1)//2, -1)

    return impulse_response


def apply_dynamic_window_to_impulse_response(impulse_response,  # B, n_frames, 2*(n_mag-1) or 2*n_mag-1
                                             half_width_frames):        # B，n_frames, 1
    ir_size = int(impulse_response.size(-1)) # 2*(n_mag -1) or 2*n_mag-1
    
    window = torch.arange(-(ir_size // 2), (ir_size + 1) // 2).to(impulse_response) / half_width_frames 
    window[window > 1] = 0
    window = (1 + torch.cos(np.pi * window)) / 2 # B, n_frames, 2*(n_mag -1) or 2*n_mag-1
    
    impulse_response = impulse_response.roll(ir_size // 2, -1)
    impulse_response = impulse_response * window
    
    return impulse_response
    
        
def frequency_impulse_response(magnitudes,
                               hann_window = True,
                               half_width_frames = None):
                               
    # Get the IR
    impulse_response = torch.fft.irfft(magnitudes) # B, n_frames, 2*(n_mags-1)
    
    # Window and put in causal form.
    if hann_window:
        if half_width_frames is None:
            impulse_response = apply_window_to_impulse_response(impulse_response)
        else:
            impulse_response = apply_dynamic_window_to_impulse_response(impulse_response, half_width_frames)
    else:
        impulse_response = impulse_response.roll(impulse_response.size(-1) // 2, -1)
       
    return impulse_response


def frequency_filter(audio,
                     magnitudes,
                     hann_window=True,
                     half_width_frames=None):

    impulse_response = frequency_impulse_response(magnitudes, hann_window, half_width_frames)
    
    return fft_convolve(audio, impulse_response)
    