import torch
import torchaudio
import argparse
import os

def load_and_preprocess(path, target_sr=44100, normalize=False):
    wave, sr = torchaudio.load(path)
    if sr != target_sr:
        print(f"Resampling {path} from {sr} Hz to {target_sr} Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wave = resampler(wave)
    if wave.shape[0] > 1:
        print(f"Converting {path} to mono...")
        wave = wave.mean(dim=0, keepdim=True)
    if normalize:
        wave = wave / wave.abs().max()
    return wave.unsqueeze(0)  # Add batch dimension

def main():
    parser = argparse.ArgumentParser(description="RAVE-based style transfer between two audio files.")
    parser.add_argument('--model', type=str, required=True, help='Path to the scripted RAVE model (.ts file)')
    parser.add_argument('--content', type=str, required=True, help='Path to content/source audio')
    parser.add_argument('--style', type=str, required=True, help='Path to style audio')
    parser.add_argument('--output', type=str, required=True, help='Path to save style-transferred audio')
    parser.add_argument('--normalize', action='store_true', help='Normalize both audio files (optional)')
    parser.add_argument('--alpha', type=float, default=0.3, help='Style transfer strength [0.0–1.0]')
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    rave = torch.jit.load(args.model).eval()

    print("Loading and preprocessing content audio...")
    content_wave = load_and_preprocess(args.content, normalize=args.normalize)
    print("Loading and preprocessing style audio...")
    style_wave = load_and_preprocess(args.style, normalize=args.normalize)

    # Match lengths
    min_len = min(content_wave.shape[-1], style_wave.shape[-1])
    content_wave = content_wave[:, :, :min_len]
    style_wave = style_wave[:, :, :min_len]

    print("Encoding both...")
    with torch.no_grad():
        z_content = rave.encode(content_wave)
        z_style = rave.encode(style_wave)

        print(f"Interpolating with alpha = {args.alpha}...")
        z_mixed = (1 - args.alpha) * z_content + args.alpha * z_style

        print("Decoding style-transferred audio...")
        stylized = rave.decode(z_mixed)

    stylized = stylized.squeeze(0).cpu()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Saving output to {args.output}")
    torchaudio.save(args.output, stylized, sample_rate=44100)

if __name__ == "__main__":
    main()

