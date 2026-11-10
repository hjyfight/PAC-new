"""Fetch the pinned public SD 1.5 weights into this experiment only."""
from common import RUN, SD_PATH, SD_REPO, SD_REVISION, runtime, sha, write
import time


def main():
    runtime()
    from huggingface_hub import HfApi, snapshot_download
    import requests
    for attempt in range(5):
        try:
            snapshot_download(SD_REPO, revision=SD_REVISION, local_dir=str(SD_PATH),
                      cache_dir=str(RUN / "runtime/hf/hub"), max_workers=3,
                      allow_patterns=["model_index.json", "scheduler/*.json", "tokenizer/*",
                                      "text_encoder/config.json", "text_encoder/model.safetensors",
                                      "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
                                      "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
                                      "feature_extractor/*.json"])
            break
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            if attempt == 4: raise
            print(f"Download interrupted; resuming existing files ({attempt + 1}/4): {type(exc).__name__}", flush=True)
            time.sleep(3)
    files = {str(p.relative_to(SD_PATH)): sha(p) for p in sorted(SD_PATH.rglob("*"))
             if p.is_file() and not any(part.startswith(".") for part in p.relative_to(SD_PATH).parts)}
    remote = HfApi().model_info(SD_REPO, revision=SD_REVISION, files_metadata=True)
    expected = {item.rfilename: item.lfs["sha256"] for item in remote.siblings
                if item.rfilename in files and item.lfs}
    for name, digest in expected.items():
        if files[name] != digest: raise RuntimeError(f"Upstream LFS hash mismatch: {name}")
    write(RUN / "evidence/sd15_provenance.json", {
        "repository": SD_REPO, "revision": SD_REVISION, "files": files,
        "upstream_lfs_sha256": expected,
        "original_repository": "runwayml/stable-diffusion-v1-5",
        "note": "Original repository redirects to this public SD 1.5 repository at the same revision.",
    })
    print("SD 1.5 download and SHA256 inventory complete", flush=True)


if __name__ == "__main__":
    main()
