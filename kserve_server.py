# kserve_server.py
"""
KServe 모델 서버 — M1 Apple Silicon GPU (CoreML EP)

Triton 없이 KServe가 직접 ONNX Runtime으로 추론.
CoreMLExecutionProvider로 M1 Neural Engine / GPU 가속.

아키텍처:
  Client --HTTP:8080--> KServe Model (ONNX Runtime + CoreML)
                        (토크나이징 + 추론 + 응답)

비교:
  inference/test_kserve.py  → Transformer → Triton(gRPC) 위임
  m1-inference/kserve_server.py → 자체 추론 (predict() 오버라이드)

실행:
  python kserve_server.py
  python kserve_server.py --model_path ../output/model_optimized_fp16.onnx --http_port 8080
"""

import argparse
import asyncio
import time
from typing import Dict, Union
from pathlib import Path

import numpy as np
import onnxruntime as ort
import kserve
from kserve import InferRequest, InferResponse, InferOutput
from transformers import AutoTokenizer

DEFAULT_MODEL_PATH = str(
    Path(__file__).parent.parent / "output" / "model_optimized_fp16.onnx"
)
DEFAULT_TOKENIZER_PATH = str(
    Path(__file__).parent.parent / "output" / "tokenizer_files"
)


class RuriEmbeddingModel(kserve.Model):
    """
    KServe Model: 텍스트 → 토크나이징 → ONNX 추론 (CoreML GPU) → 임베딩 반환

    predict()를 오버라이드하여 ONNX Runtime 직접 호출.
    Triton이나 predictor_host가 필요 없음.
    """

    def __init__(
        self,
        name: str,
        model_path: str,
        tokenizer_path: str,
        max_length: int = 128,
    ):
        super().__init__(name)
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.tokenizer = None
        self.session = None
        self.provider_name = None
        self.ready = False

    def load(self):
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        # ONNX Runtime 세션 (CoreML 우선, CPU 폴백)
        available = ort.get_available_providers()
        if "CoreMLExecutionProvider" in available:
            providers = [
                ("CoreMLExecutionProvider", {}),
                ("CPUExecutionProvider", {}),
            ]
            self.provider_name = "CoreML (M1 GPU/Neural Engine)"
        else:
            providers = [("CPUExecutionProvider", {})]
            self.provider_name = "CPU"

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        actual = self.session.get_providers()

        self.ready = True
        model_mb = Path(self.model_path).stat().st_size / 1e6
        print(f"  Model:     {self.model_path} ({model_mb:.0f} MB)")
        print(f"  Tokenizer: {self.tokenizer_path}")
        print(f"  Provider:  {actual[0]} (요청: {self.provider_name})")
        print(f"  Max Length: {self.max_length}")

    async def predict(
        self,
        payload: Union[Dict, InferRequest],
        headers: Dict[str, str] = None,
        response_headers: Dict[str, str] = None,
    ) -> InferResponse:
        # 1. 텍스트 추출 + 토크나이징
        raw_data = payload.inputs[0].data
        texts = [t if isinstance(t, str) else t.decode("utf-8") for t in raw_data]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)

        # 2. ONNX Runtime 추론 (동기 호출 → run_in_executor로 비동기 래핑)
        loop = asyncio.get_event_loop()
        start = time.perf_counter()
        outputs = await loop.run_in_executor(
            None,
            lambda: self.session.run(
                None,
                {"input_ids": input_ids, "attention_mask": attention_mask},
            ),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        embeddings = outputs[0]  # (batch, 768)

        print(
            f"  Inference: {len(texts)} texts, "
            f"shape={embeddings.shape}, "
            f"{elapsed_ms:.1f}ms"
        )

        # 3. InferResponse 반환
        return InferResponse(
            response_id=str(id(payload)),
            model_name=self.name,
            infer_outputs=[
                InferOutput(
                    name="sentence_embedding",
                    shape=list(embeddings.shape),
                    datatype="FP32",
                    data=embeddings.flatten().tolist(),
                )
            ],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KServe + ONNX Runtime (M1 CoreML)")
    parser.add_argument("--model_name", default="ruri_v3")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--http_port", type=int, default=8080)
    args = parser.parse_args()

    # 모델 파일 존재 확인
    if not Path(args.model_path).exists():
        fallback = str(Path(__file__).parent.parent / "output" / "model.onnx")
        if Path(fallback).exists():
            print(f"  [WARN] {args.model_path} 없음, 폴백: {fallback}")
            args.model_path = fallback
        else:
            print(f"  [ERROR] ONNX 모델 없음: {args.model_path}")
            print(f"          먼저 실행: python export_onnx.py")
            exit(1)

    print("=" * 60)
    print("  KServe + ONNX Runtime (M1 Apple Silicon)")
    print("=" * 60)

    model = RuriEmbeddingModel(
        name=args.model_name,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
    )
    model.load()

    kserve.ModelServer(
        http_port=args.http_port,
        workers=1,
    ).start([model])
