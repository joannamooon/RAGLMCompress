import logging
import numpy as np
import os
import time
import wave
from glob import glob
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import WavLMModel

from arithmetic_coder import ac_utils, arithmetic_coder
from evaluation.LLMCompress import Metric

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ==================== Configuration ====================
class WavLMCompressionConfig:
    """Configuration for WavLM-based lossless audio compression"""

    # WavLM encoder
    WAVLM_MODEL_NAME = "microsoft/wavlm-large"
    # Layer to extract features from.
    # wavlm-large has 24 transformer layers (hidden_states indices 0-24).
    # - Layer 6  (25% depth): phonetic features — good for unit discovery, weak context
    # - Layer 18 (75% depth): semantic/linguistic features — best for compression context
    # Treat as a hyperparameter; higher layers give richer context at the cost of
    # fuzzier representations. 18 is the recommended starting point for wavlm-large.
    # For wavlm-base (12 layers), use layer 9 instead.
    WAVLM_LAYER = 18
    WAVLM_HIDDEN_SIZE = 1024  # wavlm-large hidden dim (768 for wavlm-base)

    # Causal decoder
    DECODER_HIDDEN_SIZE = 512
    DECODER_NUM_LAYERS = 6
    DECODER_NUM_HEADS = 8
    DECODER_DROPOUT = 0.1

    # LoRA (applied to WavLM attention projections)
    LORA_R = 8
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]

    # Audio — must match source files (LibriSpeech is 16kHz 16-bit mono)
    SAMPLE_RATE = 16000
    BYTES_PER_SAMPLE = 2           # 16-bit PCM
    SAMPLES_PER_FRAME = 320        # WavLM frame size at 16kHz (20ms)
    BYTES_PER_FRAME = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE  # 640 bytes/frame
    CONTEXT_FRAMES = 50            # ~1 second of past context fed to WavLM
    MIN_CONTEXT_SAMPLES = 400      # WavLM minimum input length

    # Training
    TRAIN_DATA_PATH = "datasets/LibriSpeech/dev-clean"
    CHECKPOINT_PATH = "pretrained/wavlm_compressor/checkpoint.pth"
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10

    # Dataset paths
    DATASET_AUDIO = "datasets/librispeech/wav/*.wav"
    TEST_DATASET_AUDIO = "datasets/test_workflow/wav/*.wav"

    # Output
    COMPRESSED_OUTPUT = "compressed_audio.bin"
    PRECISION = 64

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== Causal Decoder ====================
class AudioCausalDecoder(nn.Module):
    """
    Causal transformer decoder for byte-level audio prediction.

    Self-attends over past bytes (causal mask) and cross-attends over
    WavLM context features to predict P(next byte | past bytes, WavLM context).

    Vocabulary: 256 byte values (0-255) + 1 start token (256).
    """

    def __init__(
        self,
        wavlm_hidden_size: int = WavLMCompressionConfig.WAVLM_HIDDEN_SIZE,
        hidden_size: int = WavLMCompressionConfig.DECODER_HIDDEN_SIZE,
        num_layers: int = WavLMCompressionConfig.DECODER_NUM_LAYERS,
        num_heads: int = WavLMCompressionConfig.DECODER_NUM_HEADS,
        dropout: float = WavLMCompressionConfig.DECODER_DROPOUT,
        max_seq_len: int = WavLMCompressionConfig.BYTES_PER_FRAME + 1,
    ):
        super().__init__()

        # Project WavLM features to decoder hidden size
        self.context_proj = nn.Linear(wavlm_hidden_size, hidden_size)

        # Byte embedding: 256 byte values + start token (id=256)
        self.byte_embed = nn.Embedding(257, hidden_size)

        # Positional embedding over the byte sequence within a frame
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)

        # Transformer decoder:
        #   - self-attention (causal): over past byte positions
        #   - cross-attention: over WavLM context features
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output head: hidden → logits over 256 byte values
        self.out = nn.Linear(hidden_size, 256)

    def forward(
        self,
        byte_ids: torch.Tensor,           # [B, T] — byte values with start token prepended
        context_features: torch.Tensor,   # [B, T_frames, wavlm_hidden]
        memory_key_padding_mask: Optional[torch.Tensor] = None,  # [B, T_frames]
    ) -> torch.Tensor:
        """
        :return: logits over 256 byte values [B, T, 256]
        """
        T = byte_ids.shape[1]

        positions = torch.arange(T, device=byte_ids.device).unsqueeze(0)
        x = self.byte_embed(byte_ids) + self.pos_embed(positions)  # [B, T, hidden]

        memory = self.context_proj(context_features)  # [B, T_frames, hidden]

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=byte_ids.device
        )

        out = self.transformer(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out(out)  # [B, T, 256]


# ==================== WavLM Context Encoder ====================
class WavLMContextEncoder(nn.Module):
    """
    Wraps WavLM-Large for audio context encoding.
    Extracts hidden states from a chosen layer as frame-level features.
    Optionally applies LoRA to WavLM attention projections.
    """

    def __init__(
        self,
        model_name: str = WavLMCompressionConfig.WAVLM_MODEL_NAME,
        layer: int = WavLMCompressionConfig.WAVLM_LAYER,
        use_lora: bool = False,
    ):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.layer = layer
        self.use_lora = use_lora

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=WavLMCompressionConfig.LORA_R,
                    lora_alpha=WavLMCompressionConfig.LORA_ALPHA,
                    target_modules=WavLMCompressionConfig.LORA_TARGET_MODULES,
                    lora_dropout=WavLMCompressionConfig.LORA_DROPOUT,
                )
                self.wavlm = get_peft_model(self.wavlm, lora_config)
                print("LoRA applied to WavLM.")
            except ImportError:
                print("peft not installed — running WavLM without LoRA.")
                self.use_lora = False

    def forward(
        self,
        waveform: torch.Tensor,                        # [B, T_samples] float32, normalized
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :return: hidden states from target layer [B, T_frames, hidden_size]
        """
        outputs = self.wavlm(
            waveform,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs.hidden_states[self.layer]  # [B, T_frames, hidden_size]


# ==================== WavLM Compressor ====================
class WavLMCompressor(nn.Module):
    """
    Full audio compressor: WavLM context encoder + causal byte decoder.

    For each audio frame, WavLM encodes all past frames into context features.
    The causal decoder predicts P(next byte | past bytes in frame, WavLM context),
    which is used as the probability model for arithmetic coding.
    """

    def __init__(self, use_lora: bool = False):
        super().__init__()
        self.encoder = WavLMContextEncoder(use_lora=use_lora)
        self.decoder = AudioCausalDecoder()

    def forward(
        self,
        context_waveform: torch.Tensor,  # [B, T_samples] — past audio frames
        frame_bytes: torch.Tensor,       # [B, T_bytes] — current frame bytes (start token prepended)
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :return: logits [B, T_bytes, 256]
        """
        context_features = self.encoder(context_waveform)
        return self.decoder(frame_bytes, context_features, memory_key_padding_mask)


# ==================== Audio I/O ====================
def load_audio_as_bytes(file_path: str) -> Tuple[bytes, dict]:
    """
    Load a WAV file as raw PCM bytes.

    :param file_path: path to WAV file
    :return: (raw PCM bytes, wav params dict)
    """
    with wave.open(file_path, "rb") as wav:
        params = wav.getparams()
        frames = wav.readframes(params.nframes)
    wav_params = {
        "nchannels": params.nchannels,
        "sampwidth": params.sampwidth,
        "framerate": params.framerate,
        "nframes": params.nframes,
        "comptype": params.comptype,
        "compname": params.compname,
    }
    return frames, wav_params


def bytes_to_waveform(raw_bytes: bytes, sample_width: int = 2) -> np.ndarray:
    """
    Convert raw PCM bytes to float32 waveform normalized to [-1, 1].

    :param raw_bytes: raw PCM bytes
    :param sample_width: bytes per sample (2 = 16-bit, 1 = 8-bit)
    :return: float32 waveform
    """
    if sample_width == 2:
        samples = np.frombuffer(raw_bytes, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0
    elif sample_width == 1:
        samples = np.frombuffer(raw_bytes, dtype=np.uint8)
        return (samples.astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")


def write_audio_compressed(
    filename: str, data: bytes, num_padded_bits: int, original_length: int
):
    """
    Write compressed audio data to file.

    File format:
      - 1 byte:  num_padded_bits
      - 4 bytes: original_length in bytes (supports files up to ~4GB)
      - rest:    compressed data

    :param filename: output file path
    :param data: compressed bytes
    :param num_padded_bits: number of padded bits (0-7)
    :param original_length: original audio length in bytes
    """
    if not 0 <= num_padded_bits <= 7:
        raise ValueError("num_padded_bits must be between 0 and 7.")
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes.")

    out_dir = os.path.dirname(filename)
    os.makedirs(out_dir or ".", exist_ok=True)

    with open(filename, "wb") as f:
        f.write(num_padded_bits.to_bytes(1, "big"))
        f.write(original_length.to_bytes(4, "big"))
        f.write(data)


def read_audio_compressed(filename: str) -> Tuple[bytes, int, int]:
    """
    Read compressed audio data from file.

    :param filename: input file path
    :return: (compressed bytes, num_padded_bits, original_length)
    """
    with open(filename, "rb") as f:
        padding_byte = f.read(1)
        if not padding_byte:
            raise EOFError("File is empty or improperly formatted.")
        original_length_bytes = f.read(4)
        if not original_length_bytes:
            raise EOFError("File is improperly formatted: missing original length.")
        num_padded_bits = int.from_bytes(padding_byte, "big")
        original_length = int.from_bytes(original_length_bytes, "big")
        data = f.read()
    return data, num_padded_bits, original_length


# ==================== Context Utilities ====================
def _empty_context(device: torch.device) -> torch.Tensor:
    """Return a single zero-frame placeholder when no past context exists."""
    return torch.zeros(
        1, 1, WavLMCompressionConfig.WAVLM_HIDDEN_SIZE, device=device
    )


def _get_context_features(
    waveform: np.ndarray,
    frame_idx: int,
    model: WavLMCompressor,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract WavLM context features from past frames.

    Uses frames 0..frame_idx-1 as context (causal — no future frames).

    :param waveform: float32 waveform available so far
    :param frame_idx: index of the frame currently being compressed/decoded
    :param model: WavLMCompressor
    :param device: torch device
    :return: context features [1, T_frames, hidden_size]
    """
    if frame_idx == 0:
        return _empty_context(device)

    samples_per_frame = WavLMCompressionConfig.SAMPLES_PER_FRAME
    context_frames = WavLMCompressionConfig.CONTEXT_FRAMES
    min_samples = WavLMCompressionConfig.MIN_CONTEXT_SAMPLES

    ctx_start = max(0, frame_idx - context_frames) * samples_per_frame
    ctx_end = frame_idx * samples_per_frame
    ctx_audio = waveform[ctx_start:ctx_end]

    # WavLM requires a minimum input length — pad if needed
    if len(ctx_audio) < min_samples:
        ctx_audio = np.pad(ctx_audio, (0, min_samples - len(ctx_audio)))

    context_waveform = (
        torch.tensor(ctx_audio, dtype=torch.float32).unsqueeze(0).to(device)
    )
    return model.encoder(context_waveform)  # [1, T_frames, hidden_size]


# ==================== Training ====================
def train(
    model: WavLMCompressor,
    train_files: List[str],
    device: torch.device,
    num_epochs: int = None,
    learning_rate: float = None,
    checkpoint_path: str = None,
):
    """
    Train the causal decoder (and optionally WavLM LoRA weights).

    WavLM base weights are always frozen. Only the causal decoder and
    any LoRA adapter weights are updated.

    :param model: WavLMCompressor
    :param train_files: list of WAV file paths
    :param device: torch device
    :param num_epochs: training epochs
    :param learning_rate: optimizer learning rate
    :param checkpoint_path: path to save checkpoints
    """
    if num_epochs is None:
        num_epochs = WavLMCompressionConfig.NUM_EPOCHS
    if learning_rate is None:
        learning_rate = WavLMCompressionConfig.LEARNING_RATE
    if checkpoint_path is None:
        checkpoint_path = WavLMCompressionConfig.CHECKPOINT_PATH

    # Freeze WavLM base; allow LoRA adapter weights if enabled
    for name, param in model.encoder.wavlm.named_parameters():
        param.requires_grad = "lora" in name if model.encoder.use_lora else False

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    bytes_per_frame = WavLMCompressionConfig.BYTES_PER_FRAME
    samples_per_frame = WavLMCompressionConfig.SAMPLES_PER_FRAME
    start_token = 256

    print(f"Training on {len(train_files)} file(s) for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # WavLM stays in eval — we only update the decoder
        model.train()
        model.encoder.wavlm.eval()

        epoch_loss = 0.0
        num_steps = 0

        for file_path in train_files:
            try:
                raw_bytes, wav_params = load_audio_as_bytes(file_path)
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
                continue

            waveform = bytes_to_waveform(raw_bytes, wav_params["sampwidth"])
            total_frames = len(waveform) // samples_per_frame

            if total_frames < 2:
                continue

            for frame_idx in range(1, total_frames):
                byte_start = frame_idx * bytes_per_frame
                byte_end = byte_start + bytes_per_frame
                if byte_end > len(raw_bytes):
                    break

                frame_byte_vals = list(raw_bytes[byte_start:byte_end])

                # WavLM is frozen — no gradients through encoder
                with torch.no_grad():
                    context_features = _get_context_features(
                        waveform, frame_idx, model, device
                    )

                # Input: [start_token, byte_0, ..., byte_{N-2}]
                input_ids = torch.tensor(
                    [start_token] + frame_byte_vals[:-1], dtype=torch.long
                ).unsqueeze(0).to(device)

                target_ids = torch.tensor(
                    frame_byte_vals, dtype=torch.long
                ).unsqueeze(0).to(device)

                logits = model.decoder(input_ids, context_features)  # [1, T, 256]

                loss = criterion(logits.view(-1, 256), target_ids.view(-1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_steps += 1

        avg_loss = epoch_loss / max(num_steps, 1)
        print(f"Epoch {epoch + 1}/{num_epochs} — avg loss: {avg_loss:.4f}")

        out_dir = os.path.dirname(checkpoint_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")


# ==================== Compression ====================
def wavlm_compress(
    raw_bytes: bytes,
    waveform: np.ndarray,
    model: WavLMCompressor,
    metric: Metric,
    device: torch.device,
    precision: int = None,
) -> Tuple[bytes, int, int]:
    """
    Losslessly compress raw PCM audio bytes using WavLM + causal decoder.

    For each frame, WavLM encodes past context and the causal decoder produces
    P(next byte) via teacher forcing. Arithmetic coding encodes actual byte
    values losslessly.

    :param raw_bytes: raw PCM bytes to compress
    :param waveform: float32 waveform of the full audio (for WavLM context)
    :param model: trained WavLMCompressor
    :param metric: compression metric tracker
    :param device: torch device
    :param precision: arithmetic coding precision
    :return: (compressed_bytes, num_padded_bits, original_length_in_bytes)
    """
    if precision is None:
        precision = WavLMCompressionConfig.PRECISION

    bytes_per_frame = WavLMCompressionConfig.BYTES_PER_FRAME
    samples_per_frame = WavLMCompressionConfig.SAMPLES_PER_FRAME
    start_token = 256

    # Trim to frame-aligned length
    total_frames = len(waveform) // samples_per_frame
    aligned_length = total_frames * bytes_per_frame
    raw_bytes = raw_bytes[:aligned_length]

    output = []
    ac_encoder = arithmetic_coder.Encoder(
        base=2,
        precision=precision,
        output_fn=output.append,
    )

    model.eval()
    with torch.inference_mode():
        for frame_idx in range(total_frames):
            context_features = _get_context_features(
                waveform, frame_idx, model, device
            )

            byte_start = frame_idx * bytes_per_frame
            frame_byte_vals = list(raw_bytes[byte_start : byte_start + bytes_per_frame])

            # Teacher forcing: compute all logits for this frame in one pass
            input_ids = torch.tensor(
                [start_token] + frame_byte_vals[:-1], dtype=torch.long
            ).unsqueeze(0).to(device)

            logits = model.decoder(input_ids, context_features)  # [1, bytes_per_frame, 256]
            probs = logits.softmax(dim=-1).squeeze(0).cpu().numpy()  # [bytes_per_frame, 256]

            for byte_val, prob in zip(frame_byte_vals, probs):
                ac_encoder.encode(
                    ac_utils.normalize_pdf_for_arithmetic_coding(prob, np.float32),
                    byte_val,
                )

    ac_encoder.terminate()

    compressed_bits = "".join(map(str, output))
    compressed_bytes, num_padded_bits = ac_utils.bits_to_bytes(compressed_bits)

    metric.accumulate(len(compressed_bytes), aligned_length)
    compress_rate, compress_ratio = metric.compute_ratio()

    logger.info(f"compressed length: {metric.compressed_length} bytes")
    logger.info(f"original length:   {metric.total_length} bytes")
    logger.info(f"compression ratio: {compress_ratio:.6f}")
    logger.info(f"compression rate:  {compress_rate:.6f}x")

    return compressed_bytes, num_padded_bits, aligned_length


# ==================== Decompression ====================
def wavlm_decode(
    compressed_bytes: bytes,
    num_padded_bits: int,
    original_length: int,
    model: WavLMCompressor,
    device: torch.device,
    precision: int = None,
) -> bytes:
    """
    Losslessly decompress audio bytes using WavLM + causal decoder.

    Builds the waveform incrementally from decoded bytes so WavLM context
    is derived only from already-decoded audio (no original waveform needed).

    :param compressed_bytes: compressed audio data
    :param num_padded_bits: padding bits from compression
    :param original_length: original audio length in bytes
    :param model: trained WavLMCompressor (must be same as used for compression)
    :param device: torch device
    :param precision: arithmetic coding precision
    :return: decompressed raw PCM bytes
    """
    if precision is None:
        precision = WavLMCompressionConfig.PRECISION

    bytes_per_frame = WavLMCompressionConfig.BYTES_PER_FRAME
    start_token = 256

    data_iter = iter(
        ac_utils.bytes_to_bits(compressed_bytes, num_padded_bits=num_padded_bits)
    )

    def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
        try:
            return int(next(bit_sequence))
        except StopIteration:
            return None

    ac_decoder = arithmetic_coder.Decoder(
        base=2,
        precision=precision,
        input_fn=_input_fn,
    )

    total_frames = original_length // bytes_per_frame
    decoded_bytes: List[int] = []
    decoded_waveform = np.array([], dtype=np.float32)

    model.eval()
    with torch.inference_mode():
        for frame_idx in range(total_frames):
            # Context built from already-decoded audio
            context_features = _get_context_features(
                decoded_waveform, frame_idx, model, device
            )

            # Decode bytes one at a time (autoregressive)
            frame_decoded: List[int] = []
            current_input = [start_token]

            for _ in range(bytes_per_frame):
                input_tensor = (
                    torch.tensor(current_input, dtype=torch.long)
                    .unsqueeze(0)
                    .to(device)
                )
                logits = model.decoder(input_tensor, context_features)
                prob = logits[0, -1].softmax(dim=-1).cpu().numpy()  # [256]

                decoded_byte = ac_decoder.decode(
                    ac_utils.normalize_pdf_for_arithmetic_coding(prob, np.float32)
                )
                frame_decoded.append(decoded_byte)
                current_input.append(decoded_byte)

            decoded_bytes.extend(frame_decoded)

            # Extend decoded waveform for next frame's context
            frame_waveform = bytes_to_waveform(
                bytes(frame_decoded), WavLMCompressionConfig.BYTES_PER_SAMPLE
            )
            decoded_waveform = np.concatenate([decoded_waveform, frame_waveform])

    return bytes(decoded_bytes)


# ==================== Model Loading ====================
def load_model(
    checkpoint_path: str = None,
    use_lora: bool = False,
    device: torch.device = None,
) -> WavLMCompressor:
    """
    Load WavLMCompressor from checkpoint or initialise a fresh model.

    :param checkpoint_path: path to saved checkpoint (None = fresh model)
    :param use_lora: whether to apply LoRA to WavLM
    :param device: torch device
    :return: WavLMCompressor in eval mode
    """
    if device is None:
        device = torch.device(WavLMCompressionConfig.DEVICE)

    print("Loading WavLM compressor...")
    model = WavLMCompressor(use_lora=use_lora)

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint found — using untrained model.")

    return model.to(device).eval()


# ==================== Test Functions ====================
def test_training():
    """Train the causal decoder on available WAV files."""
    print("\n" + "=" * 60)
    print("=== Testing Training ===")
    print("=" * 60)

    device = torch.device(WavLMCompressionConfig.DEVICE)
    model = WavLMCompressor(use_lora=False)

    train_files = glob(WavLMCompressionConfig.TEST_DATASET_AUDIO)
    if not train_files:
        print(f"No training files at {WavLMCompressionConfig.TEST_DATASET_AUDIO}")
        return

    print(f"Found {len(train_files)} training file(s).")
    train(
        model=model,
        train_files=train_files,
        device=device,
        num_epochs=2,
        checkpoint_path=WavLMCompressionConfig.CHECKPOINT_PATH,
    )


def test_workflow():
    """Test full compression → decompression → lossless verification."""
    print("\n" + "=" * 60)
    print("=== Testing WavLM Compression Workflow ===")
    print("=" * 60)

    device = torch.device(WavLMCompressionConfig.DEVICE)
    model = load_model(
        checkpoint_path=WavLMCompressionConfig.CHECKPOINT_PATH,
        device=device,
    )

    test_files = glob(WavLMCompressionConfig.TEST_DATASET_AUDIO)
    if not test_files:
        print(f"No test files at {WavLMCompressionConfig.TEST_DATASET_AUDIO}")
        return

    for wav_file in test_files:
        print(f"\nProcessing: {os.path.basename(wav_file)}")
        print("-" * 60)

        raw_bytes, wav_params = load_audio_as_bytes(wav_file)
        waveform = bytes_to_waveform(raw_bytes, wav_params["sampwidth"])

        print(f"Duration:  {len(waveform) / wav_params['framerate']:.2f}s")
        print(f"Raw bytes: {len(raw_bytes):,}")

        metric = Metric()

        # --- Compress ---
        t0 = time.time()
        compressed_bytes, num_padded_bits, original_length = wavlm_compress(
            raw_bytes, waveform, model, metric, device
        )
        t1 = time.time()

        write_audio_compressed(
            WavLMCompressionConfig.COMPRESSED_OUTPUT,
            compressed_bytes,
            num_padded_bits,
            original_length,
        )

        compress_rate, compress_ratio = metric.compute_ratio()
        print(f"Compressed:        {len(compressed_bytes):,} bytes")
        print(f"Compression ratio: {compress_ratio:.4f}")
        print(f"Compression rate:  {compress_rate:.4f}x")
        print(f"Compression time:  {t1 - t0:.2f}s")

        # --- Decompress ---
        compressed_bytes, num_padded_bits, original_length = read_audio_compressed(
            WavLMCompressionConfig.COMPRESSED_OUTPUT
        )

        t2 = time.time()
        decompressed_bytes = wavlm_decode(
            compressed_bytes, num_padded_bits, original_length, model, device
        )
        t3 = time.time()
        print(f"Decompression time: {t3 - t2:.2f}s")

        # --- Verify lossless ---
        aligned_length = (
            (len(waveform) // WavLMCompressionConfig.SAMPLES_PER_FRAME)
            * WavLMCompressionConfig.BYTES_PER_FRAME
        )
        original_trimmed = raw_bytes[:aligned_length]

        if original_trimmed == decompressed_bytes:
            print("✓ Lossless verification passed!")
        else:
            print("✗ Lossless verification FAILED")
            print(f"  Original:     {len(original_trimmed):,} bytes")
            print(f"  Decompressed: {len(decompressed_bytes):,} bytes")


# ==================== Main ====================
if __name__ == "__main__":
    # Phase 1: train the causal decoder
    # test_training()

    # Phase 2: test compression/decompression workflow
    test_workflow()
