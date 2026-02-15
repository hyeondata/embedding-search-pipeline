"""KServe V1/V2 프로토콜 임베딩 클라이언트"""

from __future__ import annotations

import numpy as np
import requests

# ── ruri-v3 비대칭 검색 프리픽스 ──────────────────
RURI_QUERY_PREFIX = "検索クエリ: "
RURI_DOCUMENT_PREFIX = "検索文書: "
RURI_ENCODE_PREFIX = ""

# ── 지원 프로토콜 ──────────────────────────────────
SUPPORTED_PROTOCOLS = ("v1", "v2")


class EmbeddingClient:
    """
    KServe V1/V2 Inference Protocol로 텍스트 임베딩을 요청.

    requests.Session으로 HTTP 커넥션 풀링 → 멀티스레드에서 효율적.

    Args:
        base_url:   KServe 서비스 URL (예: "http://localhost:8080")
        model_name: 모델 이름 (예: "ruri_v3")
        protocol:   "v2" (기본) 또는 "v1"
        timeout:    HTTP 요청 타임아웃 (초)

    프로토콜 차이:
        V2 (Open Inference Protocol):
          - URL: /v2/models/{name}/infer
          - 요청: {"inputs": [{"name":"text", "shape":[N], "datatype":"BYTES", "data":[...]}]}
          - 응답: {"outputs": [{"data":[...], "shape":[N, dim]}]}

        V1 (TFServing 호환):
          - URL: /v1/models/{name}:predict
          - 요청: {"instances": [{"text": "..."}, ...]}
          - 응답: {"predictions": [[0.1, 0.2, ...], ...]}
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        protocol: str = "v2",
        timeout: int = 120,
    ):
        if protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(
                f"지원하지 않는 프로토콜: {protocol!r} "
                f"(지원: {', '.join(SUPPORTED_PROTOCOLS)})"
            )

        base = base_url.rstrip("/")
        self.protocol = protocol
        self.timeout = timeout
        self.session = requests.Session()

        if protocol == "v2":
            self.url = f"{base}/v2/models/{model_name}/infer"
        else:
            self.url = f"{base}/v1/models/{model_name}:predict"

    def _build_payload(self, texts: list[str]) -> dict:
        """프로토콜에 맞는 요청 payload 생성."""
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

    def _parse_response(self, body: dict) -> np.ndarray:
        """프로토콜에 맞게 응답 파싱 → (N, dim) ndarray."""
        if self.protocol == "v2":
            output = body["outputs"][0]
            return np.array(output["data"]).reshape(output["shape"])
        else:
            return np.array(body["predictions"])

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        텍스트 리스트 → (N, dim) 임베딩 배열

        V2: data는 flat 배열 + shape으로 차원 복원.
        V1: predictions는 이미 2D 배열.
        """
        payload = self._build_payload(texts)
        resp = self.session.post(self.url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return self._parse_response(resp.json())

    def embed_with_prefix(
        self, texts: list[str], prefix: str = RURI_QUERY_PREFIX
    ) -> np.ndarray:
        """
        프리픽스를 자동으로 추가하여 임베딩.

        ruri-v3 모델은 비대칭 임베딩을 위해 프리픽스가 필요:
          - 검색 쿼리: RURI_QUERY_PREFIX ("検索クエリ: ")
          - 검색 문서: RURI_DOCUMENT_PREFIX ("検索文書: ")
          - 의미적 인코딩: RURI_ENCODE_PREFIX ("")

        Args:
            texts:  텍스트 리스트
            prefix: 각 텍스트에 추가할 프리픽스 (기본: RURI_QUERY_PREFIX)

        Returns:
            (N, dim) 임베딩 배열
        """
        prefixed = [f"{prefix}{t}" for t in texts]
        return self.embed(prefixed)
