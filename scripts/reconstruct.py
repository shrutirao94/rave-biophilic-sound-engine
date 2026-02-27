import torchaudio
import torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = torch.jit.load(args.model)

    print(f"Loading input audio from {args.input}...")
    wav, sr = torchaudio.load(args.input)
    if sr != 44100:
        print("Resampling...")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)
        wav = resampler(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    wav = wav.unsqueeze(0)  # [B, C, T]

    print("Running encode/decode...")
    with torch.no_grad():
        z = model.encode(wav)
        recon = model.decode(z)


    if args.normalize:
        recon = recon / recon.abs().max()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torchaudio.save(args.output, recon.squeeze(0).cpu(), 44100)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()

