#!/usr/bin/env python3
# generate_products.py
"""
일본어 상품명 40만개 생성기

카테고리별 어휘를 조합하여 EC(전자상거래) 스타일의 상품명을 생성.
출력: data/products_400k.txt (1줄 1상품명)

실행:
  python generate_products.py
  python generate_products.py --count 400000 --output data/products_400k.txt
"""

import argparse
import random
import time
from pathlib import Path

# ============================================================
# 공통 어휘
# ============================================================

# 산지・ブランド
ORIGINS = [
    "北海道産", "京都", "博多", "讃岐", "信州", "沖縄", "静岡",
    "宇治", "鹿児島", "新潟", "青森", "長崎", "熊本", "広島",
    "仙台", "金沢", "名古屋", "神戸", "横浜", "奈良", "秋田",
    "山形", "岩手", "富山", "三重", "和歌山", "愛媛", "高知",
    "大分", "佐賀", "鳥取", "島根", "山口", "香川", "福井",
    "岡山", "岐阜", "群馬", "栃木", "茨城", "千葉", "埼玉",
]

# 品質修飾語
QUALITY = [
    "プレミアム", "特選", "本格", "厳選", "極上", "贅沢",
    "こだわりの", "老舗の", "産地直送", "朝採れ", "天然",
    "有機", "無添加", "手作り", "数量限定", "期間限定",
    "新発売", "大容量", "お徳用", "業務用", "家庭用",
    "高級", "最高級", "濃厚", "芳醇", "熟成",
]

# 用途
PURPOSES = [
    "ギフト用", "お歳暮", "お中元", "母の日", "父の日",
    "誕生日", "お祝い", "内祝い", "お返し", "手土産",
    "自分へのご褒美", "家族向け", "業務用", "お試し",
]

# ============================================================
# 食品
# ============================================================
FOOD_ITEMS = [
    "ラーメン", "うどん", "そば", "パスタ", "焼きそば",
    "餃子", "シュウマイ", "肉まん", "春巻き", "小籠包",
    "カレー", "ハヤシライス", "ビーフシチュー", "グラタン",
    "ハンバーグ", "ステーキ", "ローストビーフ", "焼肉セット",
    "唐揚げ", "とんかつ", "コロッケ", "天ぷらセット",
    "寿司セット", "刺身盛り合わせ", "海鮮丼の具", "いくら",
    "たらこ", "明太子", "うなぎ蒲焼", "鮭切り身",
    "ホタテ", "カニ", "エビ", "牡蠣", "マグロ",
    "和牛", "黒毛和牛", "豚バラ", "鶏もも肉", "ラム肉",
    "チーズ", "バター", "ヨーグルト", "生クリーム",
    "食パン", "クロワッサン", "バゲット", "メロンパン",
    "ケーキ", "チーズケーキ", "ショートケーキ", "ロールケーキ",
    "チョコレート", "マカロン", "プリン", "ゼリー",
    "大福", "どら焼き", "まんじゅう", "羊羹", "せんべい",
    "団子", "わらび餅", "抹茶スイーツ", "和菓子詰め合わせ",
    "味噌", "醤油", "ポン酢", "ドレッシング", "マヨネーズ",
    "梅干し", "漬物", "キムチ", "納豆", "豆腐",
    "米", "もち米", "雑穀米", "玄米",
]

FOOD_SPECS = [
    "3食入り", "5食入り", "8食入り", "10食セット", "12個入り",
    "20個入り", "30個入り", "50個入り", "100g", "200g",
    "300g", "500g", "1kg", "2kg", "3kg", "5kg", "10kg",
    "6パック", "12パック", "24個セット",
    "冷凍", "冷蔵", "常温保存", "真空パック",
]

FOOD_PATTERNS = [
    lambda: f"{random.choice(ORIGINS)} {random.choice(QUALITY)} {random.choice(FOOD_ITEMS)} {random.choice(FOOD_SPECS)}",
    lambda: f"{random.choice(ORIGINS)} {random.choice(FOOD_ITEMS)} {random.choice(FOOD_SPECS)}",
    lambda: f"{random.choice(QUALITY)} {random.choice(FOOD_ITEMS)} {random.choice(FOOD_SPECS)}",
    lambda: f"{random.choice(ORIGINS)} {random.choice(FOOD_ITEMS)} {random.choice(PURPOSES)}",
    lambda: f"{random.choice(QUALITY)} {random.choice(ORIGINS)} {random.choice(FOOD_ITEMS)}",
    lambda: f"{random.choice(FOOD_ITEMS)} {random.choice(FOOD_SPECS)} {random.choice(PURPOSES)}",
]

# ============================================================
# 飲料
# ============================================================
DRINKS = [
    "緑茶", "煎茶", "ほうじ茶", "玄米茶", "抹茶",
    "紅茶", "アールグレイ", "ダージリン", "ルイボスティー",
    "コーヒー豆", "ドリップコーヒー", "インスタントコーヒー", "カフェオレ",
    "ビール", "クラフトビール", "発泡酒", "ノンアルコールビール",
    "日本酒", "純米大吟醸", "焼酎", "泡盛", "梅酒",
    "赤ワイン", "白ワイン", "スパークリングワイン", "シャンパン",
    "ウイスキー", "ジン", "ウォッカ", "ラム",
    "炭酸水", "ミネラルウォーター", "スポーツドリンク",
    "野菜ジュース", "フルーツジュース", "スムージー",
    "豆乳", "アーモンドミルク", "オーツミルク",
]

DRINK_SPECS = [
    "500ml", "750ml", "1L", "1.5L", "2L",
    "350ml×24本", "500ml×24本", "330ml×6本",
    "100g", "200g", "500g", "1kg",
    "ティーバッグ 50包", "ティーバッグ 100包",
    "ドリップ 30袋", "ドリップ 50袋", "カプセル 60個",
    "飲み比べセット", "お試しセット",
]

DRINK_PATTERNS = [
    lambda: f"{random.choice(ORIGINS)} {random.choice(QUALITY)} {random.choice(DRINKS)} {random.choice(DRINK_SPECS)}",
    lambda: f"{random.choice(ORIGINS)} {random.choice(DRINKS)} {random.choice(DRINK_SPECS)}",
    lambda: f"{random.choice(QUALITY)} {random.choice(DRINKS)} {random.choice(DRINK_SPECS)}",
    lambda: f"{random.choice(DRINKS)} {random.choice(DRINK_SPECS)} {random.choice(PURPOSES)}",
    lambda: f"{random.choice(ORIGINS)} {random.choice(DRINKS)} {random.choice(PURPOSES)}",
]

# ============================================================
# 電子機器
# ============================================================
ELECTRONICS = [
    "ワイヤレスイヤホン", "Bluetoothスピーカー", "ヘッドホン",
    "モバイルバッテリー", "USB充電器", "ワイヤレス充電器",
    "スマートウォッチ", "フィットネストラッカー",
    "タブレットスタンド", "ノートPCスタンド", "モニターアーム",
    "ウェブカメラ", "マイク", "LEDリングライト",
    "外付けSSD", "USBメモリ", "SDカード", "HDDケース",
    "キーボード", "マウス", "マウスパッド", "トラックボール",
    "ゲーミングヘッドセット", "ゲームコントローラー",
    "スマホケース", "スマホフィルム", "タブレットケース",
    "電子書籍リーダー", "デジタルフォトフレーム",
    "ドライブレコーダー", "カーナビ", "車載充電器",
    "ロボット掃除機", "空気清浄機", "加湿器", "除湿機",
    "電気ケトル", "コーヒーメーカー", "トースター",
    "ヘアドライヤー", "電動歯ブラシ", "シェーバー",
    "体重計", "血圧計", "体温計",
    "LEDデスクライト", "間接照明", "センサーライト",
    "防犯カメラ", "スマートロック", "スマートプラグ",
]

ELEC_BRANDS = [
    "ProTech", "AirMax", "SmartLife", "EcoStar", "NexGen",
    "Zenith", "Orbit", "Pulse", "Versa", "Quantum",
    "TechNova", "SkyWave", "UniCore", "BlueShift", "PrimeGear",
]

ELEC_SPECS = [
    "ブラック", "ホワイト", "シルバー", "ネイビー", "レッド",
    "最新モデル", "2026年モデル", "第3世代", "Pro版", "Lite版",
    "大容量", "コンパクト", "軽量", "防水", "ノイズキャンセリング",
    "USB-C対応", "Bluetooth 5.3", "Wi-Fi 6対応",
    "省エネ", "静音設計", "日本語対応",
]

ELEC_PATTERNS = [
    lambda: f"{random.choice(ELEC_BRANDS)} {random.choice(ELECTRONICS)} {random.choice(ELEC_SPECS)}",
    lambda: f"{random.choice(ELECTRONICS)} {random.choice(ELEC_SPECS)} {random.choice(ELEC_BRANDS)}",
    lambda: f"{random.choice(QUALITY)} {random.choice(ELECTRONICS)} {random.choice(ELEC_SPECS)}",
    lambda: f"{random.choice(ELEC_BRANDS)} {random.choice(ELECTRONICS)}",
    lambda: f"{random.choice(ELECTRONICS)} {random.choice(ELEC_SPECS)}",
]

# ============================================================
# ファッション
# ============================================================
FASHION_ITEMS = [
    "Tシャツ", "ポロシャツ", "ワイシャツ", "パーカー", "トレーナー",
    "カーディガン", "ニットセーター", "ダウンジャケット", "コート",
    "デニムジャケット", "レザージャケット", "ブルゾン", "ベスト",
    "チノパン", "デニムパンツ", "スラックス", "ショートパンツ",
    "スカート", "ワンピース", "ブラウス", "チュニック",
    "スニーカー", "ブーツ", "サンダル", "革靴", "スリッパ",
    "リュック", "トートバッグ", "ショルダーバッグ", "ボディバッグ",
    "財布", "長財布", "キーケース", "名刺入れ", "パスケース",
    "ベルト", "ネクタイ", "マフラー", "ストール", "手袋",
    "帽子", "キャップ", "ニット帽", "バケットハット",
    "腕時計", "サングラス", "ネックレス", "ブレスレット", "ピアス",
    "ルームウェア", "パジャマ", "靴下セット", "インナーシャツ",
]

FASHION_BRANDS = [
    "URBAN STYLE", "NATURAL BASIC", "CASUAL LIFE", "MONO SELECT",
    "STREET EDGE", "CLASSIC LINE", "MODERN FIT", "PURE COTTON",
    "SUNNY DAY", "OUTDOOR GEAR", "ACTIVE WEAR", "COOL BREEZE",
]

FASHION_SPECS = [
    "S", "M", "L", "XL", "XXL", "フリーサイズ",
    "ブラック", "ホワイト", "ネイビー", "グレー", "ベージュ",
    "カーキ", "ブラウン", "レッド", "ブルー", "グリーン",
    "綿100%", "リネン素材", "ウール混", "ポリエステル",
    "春夏モデル", "秋冬モデル", "通年", "撥水加工",
    "メンズ", "レディース", "ユニセックス", "キッズ",
]

FASHION_PATTERNS = [
    lambda: f"{random.choice(FASHION_BRANDS)} {random.choice(FASHION_ITEMS)} {random.choice(FASHION_SPECS)}",
    lambda: f"{random.choice(QUALITY)} {random.choice(FASHION_ITEMS)} {random.choice(FASHION_SPECS)}",
    lambda: f"{random.choice(FASHION_ITEMS)} {random.choice(FASHION_SPECS)} {random.choice(FASHION_BRANDS)}",
    lambda: f"{random.choice(FASHION_BRANDS)} {random.choice(FASHION_ITEMS)}",
    lambda: f"{random.choice(FASHION_ITEMS)} {random.choice(FASHION_SPECS)} {random.choice(FASHION_SPECS)}",
]

# ============================================================
# 美容・コスメ
# ============================================================
BEAUTY_ITEMS = [
    "化粧水", "美容液", "乳液", "クリーム", "クレンジング",
    "洗顔フォーム", "フェイスマスク", "アイクリーム",
    "日焼け止め", "BBクリーム", "ファンデーション", "コンシーラー",
    "リップスティック", "リップグロス", "チーク", "アイシャドウ",
    "マスカラ", "アイライナー", "眉ペンシル",
    "シャンプー", "コンディショナー", "トリートメント", "ヘアオイル",
    "ヘアワックス", "ヘアスプレー", "カラートリートメント",
    "ボディソープ", "ボディクリーム", "ハンドクリーム",
    "ネイルカラー", "除光液", "ネイルケアセット",
    "香水", "練り香水", "ボディミスト",
    "サプリメント", "ビタミンC", "コラーゲン", "プロテイン",
]

BEAUTY_BRANDS = [
    "LUNA GLOW", "AQUA VEIL", "ROSE GARDEN", "SILK TOUCH",
    "NATURE PURE", "CRYSTAL CLEAR", "BLOOM BEAUTY", "MIST AURA",
    "PEARL SKIN", "VELVET SOFT", "HERB GARDEN", "ZEN CARE",
]

BEAUTY_SPECS = [
    "30ml", "50ml", "100ml", "150ml", "200ml", "300ml",
    "30g", "50g", "100g", "150g",
    "敏感肌用", "乾燥肌用", "脂性肌用", "エイジングケア",
    "美白", "保湿", "ハリ・弾力", "毛穴ケア",
    "医薬部外品", "オーガニック", "無香料", "低刺激",
    "詰め替え用", "トラベルセット", "お試しサイズ",
]

BEAUTY_PATTERNS = [
    lambda: f"{random.choice(BEAUTY_BRANDS)} {random.choice(BEAUTY_ITEMS)} {random.choice(BEAUTY_SPECS)}",
    lambda: f"{random.choice(QUALITY)} {random.choice(BEAUTY_ITEMS)} {random.choice(BEAUTY_SPECS)}",
    lambda: f"{random.choice(BEAUTY_ITEMS)} {random.choice(BEAUTY_SPECS)} {random.choice(BEAUTY_BRANDS)}",
    lambda: f"{random.choice(BEAUTY_BRANDS)} {random.choice(BEAUTY_ITEMS)}",
    lambda: f"{random.choice(BEAUTY_ITEMS)} {random.choice(BEAUTY_SPECS)} {random.choice(BEAUTY_SPECS)}",
]

# ============================================================
# 生活・ホーム
# ============================================================
HOME_ITEMS = [
    "タオルセット", "バスタオル", "フェイスタオル",
    "枕", "布団", "毛布", "敷きパッド", "シーツ", "枕カバー",
    "カーテン", "ラグ", "カーペット", "玄関マット",
    "収納ボックス", "衣装ケース", "本棚", "ラック",
    "食器セット", "マグカップ", "グラス", "箸セット",
    "フライパン", "鍋セット", "包丁", "まな板",
    "弁当箱", "水筒", "タンブラー", "保温ボトル",
    "ゴミ箱", "洗濯かご", "アイロン台",
    "洗剤", "柔軟剤", "食器用洗剤", "ハンドソープ",
    "除菌スプレー", "消臭剤", "芳香剤",
    "掃除機用紙パック", "フローリングシート", "スポンジ",
    "観葉植物", "フラワーポット", "園芸用土",
    "防災セット", "懐中電灯", "救急箱",
]

HOME_BRANDS = [
    "SIMPLE LIFE", "HOME COMFORT", "CLEAN LIVING", "ECO HOME",
    "DAILY BASICS", "NATURAL HOUSE", "COZY ROOM", "SMART STORAGE",
]

HOME_SPECS = [
    "3枚セット", "5枚組", "2個セット", "3個パック",
    "シングル", "セミダブル", "ダブル", "クイーン",
    "ホワイト", "ベージュ", "グレー", "ブラウン", "ピンク",
    "抗菌防臭", "速乾", "丸洗いOK", "防ダニ",
    "大容量", "コンパクト", "折りたたみ", "壁掛け",
]

HOME_PATTERNS = [
    lambda: f"{random.choice(HOME_BRANDS)} {random.choice(HOME_ITEMS)} {random.choice(HOME_SPECS)}",
    lambda: f"{random.choice(QUALITY)} {random.choice(HOME_ITEMS)} {random.choice(HOME_SPECS)}",
    lambda: f"{random.choice(HOME_ITEMS)} {random.choice(HOME_SPECS)} {random.choice(HOME_SPECS)}",
    lambda: f"{random.choice(HOME_BRANDS)} {random.choice(HOME_ITEMS)}",
    lambda: f"{random.choice(ORIGINS)} {random.choice(HOME_ITEMS)} {random.choice(HOME_SPECS)}",
]

# ============================================================
# スポーツ・アウトドア
# ============================================================
SPORTS_ITEMS = [
    "ランニングシューズ", "トレーニングシューズ", "ウォーキングシューズ",
    "ヨガマット", "ダンベル", "腹筋ローラー", "トレーニングチューブ",
    "プロテインシェーカー", "スポーツタオル", "リストバンド",
    "サイクルジャージ", "サイクルパンツ", "ヘルメット",
    "テント", "寝袋", "エアーマット", "ランタン", "クッカーセット",
    "トレッキングポール", "登山リュック", "レインウェア",
    "釣り竿", "リール", "ルアーセット", "タックルボックス",
    "ゴルフクラブ", "ゴルフボール", "ゴルフグローブ",
    "サッカーボール", "バスケットボール", "バレーボール",
    "バドミントンラケット", "テニスラケット", "卓球ラケット",
    "水着", "ゴーグル", "スイムキャップ",
    "スキーウェア", "スノーボード", "スキーゴーグル",
]

SPORTS_BRANDS = [
    "ACTIVE PRO", "SUMMIT GEAR", "TRAIL MASTER", "SPEED FORCE",
    "FIELD EDGE", "OCEAN SPORT", "MOUNTAIN PEAK", "FITNESS LAB",
]

SPORTS_SPECS = [
    "メンズ", "レディース", "ジュニア",
    "S", "M", "L", "XL", "26cm", "27cm", "28cm",
    "ブラック", "ホワイト", "レッド", "ブルー",
    "軽量", "防水", "通気性", "衝撃吸収", "滑り止め",
    "初心者向け", "中級者向け", "プロ仕様",
    "2人用", "3〜4人用", "ソロ用",
]

SPORTS_PATTERNS = [
    lambda: f"{random.choice(SPORTS_BRANDS)} {random.choice(SPORTS_ITEMS)} {random.choice(SPORTS_SPECS)}",
    lambda: f"{random.choice(QUALITY)} {random.choice(SPORTS_ITEMS)} {random.choice(SPORTS_SPECS)}",
    lambda: f"{random.choice(SPORTS_ITEMS)} {random.choice(SPORTS_SPECS)} {random.choice(SPORTS_SPECS)}",
    lambda: f"{random.choice(SPORTS_BRANDS)} {random.choice(SPORTS_ITEMS)}",
    lambda: f"{random.choice(SPORTS_ITEMS)} {random.choice(SPORTS_SPECS)} {random.choice(SPORTS_BRANDS)}",
]

# ============================================================
# 書籍・文具
# ============================================================
BOOK_GENRES = [
    "入門", "実践", "完全ガイド", "基礎から学ぶ", "プロが教える",
    "はじめての", "よくわかる", "図解", "マンガでわかる", "決定版",
]

BOOK_TOPICS = [
    "Python", "JavaScript", "React", "機械学習", "データサイエンス",
    "AWS", "Docker", "Kubernetes", "Go言語", "Rust",
    "英会話", "TOEIC", "簿記", "FP", "宅建",
    "投資", "株", "不動産", "ビジネス戦略", "マーケティング",
    "料理", "園芸", "DIY", "写真撮影", "イラスト",
    "子育て", "健康管理", "メンタルヘルス", "瞑想",
]

STATIONERY = [
    "ボールペン", "万年筆", "シャープペンシル", "マーカーセット",
    "ノート", "手帳", "スケジュール帳", "メモ帳",
    "ファイル", "クリアファイル", "バインダー",
    "付箋セット", "マスキングテープ", "クリップセット",
    "定規セット", "はさみ", "カッター", "のり",
    "ペンケース", "デスクオーガナイザー", "ブックスタンド",
]

BOOK_PATTERNS = [
    lambda: f"{random.choice(BOOK_GENRES)} {random.choice(BOOK_TOPICS)}",
    lambda: f"{random.choice(BOOK_GENRES)} {random.choice(BOOK_TOPICS)} {random.choice(['改訂版', '第2版', '第3版', '最新版', '2026年版'])}",
    lambda: f"{random.choice(QUALITY)} {random.choice(STATIONERY)} {random.choice(['ブラック', 'ブルー', 'レッド', '3色セット', '5本セット', '10冊パック', 'A4', 'B5', 'A5'])}",
    lambda: f"{random.choice(STATIONERY)} {random.choice(['ブラック', 'ブルー', 'ピンク', 'ホワイト', '3色セット', '5本入り', '10冊セット', 'A4', 'B5'])}",
    lambda: f"{random.choice(BOOK_TOPICS)} {random.choice(['ハンドブック', 'リファレンス', '問題集', 'ワークブック', 'テキスト'])}",
]

# ============================================================
# 全パターン統合
# ============================================================
ALL_PATTERNS = (
    FOOD_PATTERNS * 4       # 食品 — 出現率高め
    + DRINK_PATTERNS * 3
    + ELEC_PATTERNS * 3
    + FASHION_PATTERNS * 3
    + BEAUTY_PATTERNS * 2
    + HOME_PATTERNS * 2
    + SPORTS_PATTERNS * 2
    + BOOK_PATTERNS * 2
)


def generate_products(count: int, seed: int = 42) -> list[str]:
    """指定数のユニークな商品名を生成"""
    random.seed(seed)
    products = set()

    while len(products) < count:
        pattern = random.choice(ALL_PATTERNS)
        products.add(pattern())

    return list(products)


def main():
    parser = argparse.ArgumentParser(description="日本語商品名生成器")
    parser.add_argument("--count", type=int, default=400_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "products_400k.txt",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  日本語商品名生成: {args.count:,}件")
    print("=" * 60)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    products = generate_products(args.count, seed=args.seed)
    gen_time = time.time() - start

    print(f"\n  生成: {len(products):,}件 ({gen_time:.1f}秒)")
    print(f"  サンプル:")
    for p in products[:15]:
        print(f"    {p}")

    start = time.time()
    with open(args.output, "w", encoding="utf-8") as f:
        for p in products:
            f.write(p + "\n")
    write_time = time.time() - start

    size_mb = args.output.stat().st_size / 1e6
    print(f"\n  保存: {args.output} ({size_mb:.1f} MB, {write_time:.1f}秒)")
    print("  完了!")


if __name__ == "__main__":
    main()
