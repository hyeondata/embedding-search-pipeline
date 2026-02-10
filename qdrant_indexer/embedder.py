"""KServe V2 프로토콜 임베딩 클라이언트"""

import numpy as np
import requests


class EmbeddingClient:
    """
    KServe V2 Inference Protocol로 텍스트 임베딩을 요청.

    requests.Session으로 HTTP 커넥션 풀링 → 멀티스레드에서 효율적.
    """

    def __init__(self, base_url: str, model_name: str):
        self.url = f"{base_url}/v2/models/{model_name}/infer"
        self.session = requests.Session()

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        텍스트 리스트 → (N, 768) 임베딩 배열

        V2 프로토콜: data는 flat 배열, shape으로 차원 복원.
        """
        payload = {
            "inputs": [{
                "name": "text",
                "shape": [len(texts)],
                "datatype": "BYTES",
                "data": texts,
            }]
        }
        resp = self.session.post(self.url, json=payload, timeout=120)
        resp.raise_for_status()

        output = resp.json()["outputs"][0]
        return np.array(output["data"]).reshape(output["shape"])
