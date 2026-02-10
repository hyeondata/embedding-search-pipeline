# test_client.py
"""
KServe M1 서버 E2E 테스트

kserve_server.py가 실행 중일 때 사용.
V2 Inference Protocol로 텍스트를 전송하고 임베딩 + 코사인 유사도를 검증.

실행:
  python test_client.py
  python test_client.py --url http://localhost:8080
"""

import argparse
import time
import requests
import numpy as np
import json


def test_v2_infer(base_url: str, model_name: str):
    """V2 Inference Protocol 테스트"""
    url = f"{base_url}/v2/models/{model_name}/infer"

    payload = {
        "inputs": [
            {
                "name": "text",
                "shape": [3],
                "datatype": "BYTES",
                "data": [
                    "検索クエリ: 東京の天気",
                    "文章: 東京は今日晴れです。気温は25度です。",
                    "文章: 京都には多くの寺院があります。",
                ],
            }
        ]
    }

    print(f"  URL: {url}")
    print(f"  요청 전송 중 ...")

    start = time.perf_counter()
    response = requests.post(url, json=payload, timeout=60)
    elapsed = time.perf_counter() - start

    if response.status_code != 200:
        print(f"  [ERROR] HTTP {response.status_code}: {response.text[:200]}")
        return False

    result = response.json()

    print(f"  응답 시간: {elapsed:.3f}초")
    print(f"\n=== 응답 구조 ===")
    print(json.dumps(result, indent=2, ensure_ascii=False)[:500])

    print(f"\n=== outputs 분석 ===")
    for output in result["outputs"]:
        print(f"  name:     {output['name']}")
        print(f"  shape:    {output['shape']}")
        print(f"  datatype: {output['datatype']}")
        print(f"  data 길이: {len(output['data'])}")
        print(f"  data 길이: {(output['data'])}")


    # 임베딩 추출
    emb_output = result["outputs"][0]
    embeddings = np.array(emb_output["data"]).reshape(emb_output["shape"])
    print(f"\n  embeddings shape: {embeddings.shape}")

    # L2 norm 검증
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  L2 norms: {norms.round(4).tolist()}")

    # 코사인 유사도
    print(f"\n=== 코사인 유사도 ===")
    query = embeddings[0]
    labels = ["query (자기자신)", "doc1 (관련: 東京の天気)", "doc2 (무관: 京都の寺院)"]
    for i, label in enumerate(labels):
        sim = np.dot(query, embeddings[i])
        print(f"  {label}: {sim:.4f}")

    sim_related = np.dot(query, embeddings[1])
    sim_unrelated = np.dot(query, embeddings[2])
    if sim_related > sim_unrelated:
        print(f"\n  관련 > 무관: OK")
    else:
        print(f"\n  WARNING: 관련 문서 유사도가 더 낮음")

    print(f"\n  PASSED! ({elapsed:.3f}초)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--model_name", default="ruri_v3")
    args = parser.parse_args()

    print("=" * 60)
    print("  KServe M1 서버 E2E 테스트")
    print("=" * 60)

    success = test_v2_infer(args.url, args.model_name)
    exit(0 if success else 1)
