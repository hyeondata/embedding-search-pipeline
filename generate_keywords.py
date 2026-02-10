# generate_keywords.py
"""
일본어 키워드 40만개 생성기

카테고리별 단어를 조합하여 검색 쿼리 스타일의 키워드를 생성.
출력: data/keywords_400k.txt (1줄 1키워드)

실행:
  python generate_keywords.py
  python generate_keywords.py --count 400000 --output data/keywords_400k.txt
"""

import argparse
import random
import time
from pathlib import Path

# 지역
REGIONS = [
    "東京", "大阪", "京都", "北海道", "沖縄", "名古屋", "福岡", "札幌",
    "横浜", "神戸", "広島", "仙台", "新潟", "金沢", "長崎", "鹿児島",
    "奈良", "熊本", "岡山", "静岡", "千葉", "埼玉", "宮崎", "青森",
    "秋田", "山形", "岩手", "富山", "石川", "福井", "滋賀", "三重",
    "和歌山", "鳥取", "島根", "山口", "徳島", "香川", "愛媛", "高知",
    "佐賀", "大分", "栃木", "群馬", "茨城", "長野", "山梨", "岐阜",
]

# 음식
FOODS = [
    "ラーメン", "寿司", "天ぷら", "うどん", "そば", "焼肉", "刺身",
    "たこ焼き", "お好み焼き", "カレー", "丼", "おにぎり", "弁当",
    "味噌汁", "豆腐", "納豆", "漬物", "餃子", "唐揚げ", "とんかつ",
    "しゃぶしゃぶ", "すき焼き", "もつ鍋", "おでん", "焼き鳥",
    "パン", "ケーキ", "アイスクリーム", "チョコレート", "抹茶",
    "和菓子", "団子", "大福", "せんべい", "まんじゅう",
    "コーヒー", "紅茶", "緑茶", "ビール", "日本酒", "ワイン",
]

# 観光・場所
PLACES = [
    "駅", "空港", "ホテル", "旅館", "温泉", "神社", "寺院", "城",
    "公園", "動物園", "水族館", "美術館", "博物館", "映画館",
    "ショッピングモール", "デパート", "コンビニ", "スーパー",
    "レストラン", "カフェ", "居酒屋", "バー", "図書館", "病院",
    "学校", "大学", "会社", "銀行", "郵便局", "市役所",
    "海", "山", "川", "湖", "島", "森", "滝", "橋", "塔", "展望台",
]

# 行動・目的
ACTIONS = [
    "観光", "旅行", "グルメ", "ショッピング", "散歩", "登山",
    "キャンプ", "釣り", "サーフィン", "スキー", "花見", "紅葉",
    "祭り", "イベント", "ライブ", "コンサート", "展覧会",
    "予約", "営業時間", "アクセス", "料金", "口コミ", "ランキング",
    "おすすめ", "人気", "穴場", "格安", "無料", "割引",
    "駐車場", "送料", "配達", "宅配", "通販", "オンライン",
]

# 形容詞・修飾語
ADJECTIVES = [
    "美味しい", "安い", "高い", "近い", "遠い", "新しい", "古い",
    "大きい", "小さい", "有名な", "人気の", "おすすめの", "最新の",
    "伝統的な", "本格的な", "手作りの", "限定の", "特別な",
    "綺麗な", "静かな", "にぎやかな", "便利な", "快適な",
]

# 時間・季節
TIMES = [
    "朝", "昼", "夜", "深夜", "早朝", "週末", "平日", "祝日",
    "春", "夏", "秋", "冬", "1月", "2月", "3月", "4月",
    "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月",
    "今日", "明日", "来週", "今月", "来月", "2025年", "2026年",
]

# IT・ビジネス
TECH = [
    "プログラミング", "Python", "JavaScript", "AI", "機械学習",
    "データ分析", "クラウド", "AWS", "Docker", "Kubernetes",
    "アプリ開発", "Webデザイン", "SEO", "マーケティング",
    "副業", "転職", "資格", "勉強法", "オンライン講座",
    "スマホ", "パソコン", "タブレット", "ガジェット", "レビュー",
]

# 生活
LIFE = [
    "引っ越し", "賃貸", "マンション", "一戸建て", "リフォーム",
    "掃除", "洗濯", "料理", "レシピ", "ダイエット", "筋トレ",
    "健康", "美容", "化粧品", "ファッション", "ヘアスタイル",
    "育児", "子育て", "保育園", "学習塾", "習い事",
    "ペット", "犬", "猫", "結婚", "離婚", "保険", "年金", "税金",
]

# テンプレートパターン
PATTERNS = [
    # 地域 + 名詞
    lambda: f"{random.choice(REGIONS)} {random.choice(FOODS)}",
    lambda: f"{random.choice(REGIONS)} {random.choice(PLACES)}",
    lambda: f"{random.choice(REGIONS)} {random.choice(ACTIONS)}",
    # 地域 + 形容詞 + 名詞
    lambda: f"{random.choice(REGIONS)} {random.choice(ADJECTIVES)} {random.choice(FOODS)}",
    lambda: f"{random.choice(REGIONS)} {random.choice(ADJECTIVES)} {random.choice(PLACES)}",
    # 名詞 + 行動
    lambda: f"{random.choice(FOODS)} {random.choice(ACTIONS)}",
    lambda: f"{random.choice(PLACES)} {random.choice(ACTIONS)}",
    # 時間 + 地域 + 名詞
    lambda: f"{random.choice(TIMES)} {random.choice(REGIONS)} {random.choice(ACTIONS)}",
    # IT・ビジネス
    lambda: f"{random.choice(TECH)} {random.choice(ACTIONS)}",
    lambda: f"{random.choice(TECH)} {random.choice(ADJECTIVES)}",
    # 生活
    lambda: f"{random.choice(LIFE)} {random.choice(ACTIONS)}",
    lambda: f"{random.choice(REGIONS)} {random.choice(LIFE)}",
    # 複合パターン
    lambda: f"{random.choice(REGIONS)} {random.choice(TIMES)} {random.choice(FOODS)} {random.choice(ACTIONS)}",
    lambda: f"{random.choice(ADJECTIVES)} {random.choice(FOODS)} {random.choice(REGIONS)}",
    lambda: f"{random.choice(LIFE)} {random.choice(ADJECTIVES)} {random.choice(ACTIONS)}",
    lambda: f"{random.choice(TECH)} {random.choice(TIMES)} {random.choice(ACTIONS)}",
]


def generate_keywords(count: int, seed: int = 42) -> list[str]:
    random.seed(seed)
    keywords = set()

    while len(keywords) < count:
        pattern = random.choice(PATTERNS)
        keyword = pattern()
        keywords.add(keyword)

    return list(keywords)


def main():
    parser = argparse.ArgumentParser(description="日本語キーワード生成器")
    parser.add_argument("--count", type=int, default=400_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "keywords_400k.txt",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  日本語キーワード生成: {args.count:,}件")
    print("=" * 60)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    keywords = generate_keywords(args.count, seed=args.seed)
    gen_time = time.time() - start

    print(f"\n  生成: {len(keywords):,}件 ({gen_time:.1f}秒)")
    print(f"  サンプル:")
    for kw in keywords[:10]:
        print(f"    {kw}")

    start = time.time()
    with open(args.output, "w", encoding="utf-8") as f:
        for kw in keywords:
            f.write(kw + "\n")
    write_time = time.time() - start

    size_mb = args.output.stat().st_size / 1e6
    print(f"\n  保存: {args.output} ({size_mb:.1f} MB, {write_time:.1f}秒)")
    print(f"  完了!")


if __name__ == "__main__":
    main()
