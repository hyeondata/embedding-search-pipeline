"""KServe V1/V2 프로토콜 임베딩 클라이언트 (클라이언트 사이드 토크나이저 지원)

토크나이저 모드 (기본):
  - AutoTokenizer로 텍스트를 토큰화
  - input_ids + attention_mask를 INT64 텐서로 V2 전송
  - raw 모델 (ruri_v3 등)에 직접 추론

텍스트 모드 (tokenizer_name=None):
  - 기존 방식: 텍스트를 BYTES로 전송
  - BLS 파이프라인 (서버 사이드 토크나이저) 사용 시
"""

from __future__ import annotations

import numpy as np
import requests

# ── ruri-v3 비대칭 검색 프리픽스 ──────────────────
RURI_QUERY_PREFIX = "検索クエリ: "
RURI_DOCUMENT_PREFIX = "検索文書: "
RURI_ENCODE_PREFIX = ""

# ── 기본 토크나이저 ──────────────────────────────
DEFAULT_TOKENIZER = "cl-nagoya/ruri-v3-310m"
DEFAULT_MAX_LENGTH = 512

# ── 지원 프로토콜 ──────────────────────────────────
SUPPORTED_PROTOCOLS = ("v1", "v2")


class EmbeddingClient:
    """
    KServe V1/V2 Inference Protocol로 텍스트 임베딩을 요청.

    Args:
        base_url:       KServe 서비스 URL (예: "http://localhost:8000")
        model_name:     모델 이름 (예: "ruri_v3")
        protocol:       "v2" (기본) 또는 "v1"
        timeout:        HTTP 요청 타임아웃 (초)
        tokenizer_name: HuggingFace 토크나이저 이름 (기본: "cl-nagoya/ruri-v3-310m").
                        None이면 텍스트 BYTES 모드 (BLS 파이프라인용).
        max_length:     토크나이저 최대 시퀀스 길이 (기본: 512)
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        protocol: str = "v2",
        timeout: int = 120,
        tokenizer_name: str | None = DEFAULT_TOKENIZER,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        if protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(
                f"지원하지 않는 프로토콜: {protocol!r} "
                f"(지원: {', '.join(SUPPORTED_PROTOCOLS)})"
            )

        if tokenizer_name is not None and protocol != "v2":
            raise ValueError(
                "토크나이저 모드는 V2 프로토콜만 지원합니다. "
                "V1 사용 시 tokenizer_name=None으로 설정하세요."
            )

        base = base_url.rstrip("/")
        self.protocol = protocol
        self.timeout = timeout
        self.max_length = max_length
        self.session = requests.Session()

        if protocol == "v2":
            self.url = f"{base}/v2/models/{model_name}/infer"
        else:
            self.url = f"{base}/v1/models/{model_name}:predict"

        # 토크나이저 로드
        if tokenizer_name is not None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = None

    def _build_payload(self, texts: list[str]) -> dict:
        """프로토콜 + 토크나이저 설정에 맞는 요청 payload 생성."""
        if self.tokenizer is not None:
            return self._build_tokenized_payload(texts)

        if self.protocol == "v2":
            return {
                "inputs": [{
                    "name": "text",
                    "shape": [len(texts)],
                    "datatype": "BYTES",
                    "data": texts,
                }]
            }
        else:
            return {
                "instances": [{"text": t} for t in texts],
            }

    def _build_tokenized_payload(self, texts: list[str]) -> dict:
        """텍스트를 토큰화하여 V2 INT64 텐서 payload 생성."""
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)

        batch_size, seq_len = input_ids.shape

        return {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": [batch_size, seq_len],
                    "datatype": "INT64",
                    "data": input_ids.flatten().tolist(),
                },
                {
                    "name": "attention_mask",
                    "shape": [batch_size, seq_len],
                    "datatype": "INT64",
                    "data": attention_mask.flatten().tolist(),
                },
            ]
        }

    def _parse_response(self, body: dict) -> np.ndarray:
        """프로토콜에 맞게 응답 파싱 → (N, dim) ndarray."""
        if self.protocol == "v2":
            output = body["outputs"][0]
            return np.array(output["data"]).reshape(output["shape"])
        else:
            return np.array(body["predictions"])

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        텍스트 리스트 → (N, dim) 임베딩 배열.

        토크나이저 모드: 클라이언트에서 토큰화 후 INT64 텐서 전송.
        텍스트 모드: 텍스트를 그대로 전송 (BLS 파이프라인용).
        """
        payload = self._build_payload(texts)
        resp = self.session.post(self.url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return self._parse_response(resp.json())

    def embed_with_prefix(
        self, texts: list[str], prefix: str = RURI_QUERY_PREFIX
    ) -> np.ndarray:
        """프리픽스를 자동으로 추가하여 임베딩."""
        prefixed = [f"{prefix}{t}" for t in texts]
        return self.embed(prefixed)
