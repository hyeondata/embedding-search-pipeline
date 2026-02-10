# export_onnx.py
"""
ONNX 모델 변환 스크립트 (cl-nagoya/ruri-v3-310m)

Transformer + Mean Pooling + L2 Normalization을 하나의 ONNX 모델로 통합.
출력 모델은 텍스트 토큰 → 768차원 정규화 임베딩을 직접 반환.

출력:
  output/model.onnx           (base, ~1.2GB)
  output/tokenizer_files/     (SentencePiece 토크나이저)

실행:
  python export_onnx.py
  python export_onnx.py --output-dir ../output --force
"""

import sys
import time
import argparse
from pathlib import Path

MODEL_NAME = "cl-nagoya/ruri-v3-310m"


def main():
    parser = argparse.ArgumentParser(description="ruri-v3-310m ONNX Export")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "output",
    )
    parser.add_argument("--force", action="store_true", help="기존 모델 덮어쓰기")
    args = parser.parse_args()

    print("=" * 60)
    print("  ONNX Export (Transformer + Pooling + L2 Norm)")
    print("=" * 60)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / "model.onnx"

    if onnx_path.exists() and not args.force:
        size_mb = onnx_path.stat().st_size / 1e6
        print(f"\n  [SKIP] 이미 존재: {onnx_path} ({size_mb:.0f} MB)")
        print(f"         덮어쓰려면: python export_onnx.py --force")
        return True

    try:
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("\n  [ERROR] pip install torch transformers")
        return False

    # 1. Pooling 포함 모델 정의
    print(f"\n  [1/4] 모델 정의 ...")

    class ModernBertWithPooling(nn.Module):
        """
        ModernBERT + Mean Pooling + L2 Normalization

        입력: input_ids, attention_mask
        출력: sentence_embedding (batch, 768), L2 norm = 1.0
        """
        def __init__(self, model_name):
            super().__init__()
            self.transformer = AutoModel.from_pretrained(
                model_name,
                attn_implementation="eager",
                reference_compile=False,
            )

        def forward(self, input_ids, attention_mask):
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            hidden_state = outputs.last_hidden_state

            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask

            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return normalized

    model = ModernBertWithPooling(MODEL_NAME)
    model.eval()
    print(f"         ModernBertWithPooling 생성 완료")

    # 2. 더미 입력 생성
    print(f"\n  [2/4] 더미 입력 생성 ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dummy_inputs = tokenizer(
        ["検索クエリ: テスト"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    print(f"         shape: {dummy_inputs['input_ids'].shape}")

    # 3. ONNX Export
    print(f"\n  [3/4] ONNX Export 시작 (수 분 소요) ...")
    start = time.time()

    try:
        torch.onnx.export(
            model,
            (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["sentence_embedding"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "sentence_embedding": {0: "batch_size"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
    except Exception as e:
        print(f"  [ERROR] ONNX Export 실패: {e}")
        return False

    elapsed = time.time() - start
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"         완료: {elapsed:.1f}초, {size_mb:.0f} MB")

    # 4. 토크나이저 저장
    print(f"\n  [4/4] 토크나이저 저장 ...")
    tokenizer_path = args.output_dir / "tokenizer_files"
    tokenizer.save_pretrained(tokenizer_path)
    print(f"         {tokenizer_path}")

    print(f"\n  PASSED!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
