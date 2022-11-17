import gensim
from pykakasi import kakasi

def get_similar_words(word,n,model=None):
    if model==None:
        model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    return model.most_similar(word,topn=n)

def get_most_similar_word(start_letter, word, model=None,nearest_n=10, threshold=0.5):
    """
    Get similar words to the given word.
    """
    if model==None:
        print("loading word2vec model...")
        model = gensim.models.KeyedVectors.load_word2vec_format('./weights/word2vec/model.vec',binary=False)
        print("done.")

    # return a list of tuples (word, similarity)

    for similar_word, similarity in model.most_similar(word, topn=nearest_n):
        word = kanji_katakana_to_hiragana(word)
        if similar_word[0] == start_letter and similarity > threshold:
            most_similar_word = similar_word
            break
        else:
            most_similar_word = "分かりません... ( > < )"
        print(f"most similar word: {most_similar_word}")

    return most_similar_word

def kanji_katakana_to_hiragana(word):
    kak = kakasi()
    kak.setMode("J", "H")
    kak.setMode("K", "H")
    conv = kak.getConverter()
    return conv.do(word)

def get_label_name(index,language='ja'):
    if language == 'en':
        classes = [
    'aircraft carrier',
    'airplane',
    'alarm clock',
    'ambulance',
    'angel',
    'animal migration',
    'ant',
    'anvil',
    'apple',
    'arm',
    'asparagus',
    'axe',
    'backpack',
    'banana',
    'bandage',
    'barn',
    'baseball',
    'baseball bat',
    'basket',
    'basketball',
    'bat',
    'bathtub',
    'beach',
    'bear',
    'beard',
    'bed',
    'bee',
    'belt',
    'bench',
    'bicycle',
    'binoculars',
    'bird',
    'birthday cake',
    'blackberry',
    'blueberry',
    'book',
    'boomerang',
    'bottlecap',
    'bowtie',
    'bracelet',
    'brain',
    'bread',
    'bridge',
    'broccoli',
    'broom',
    'bucket',
    'bulldozer',
    'bus',
    'bush',
    'butterfly',
    'cactus',
    'cake',
    'calculator',
    'calendar',
    'camel',
    'camera',
    'camouflage',
    'campfire',
    'candle',
    'cannon',
    'canoe',
    'car',
    'carrot',
    'castle',
    'cat',
    'ceiling fan',
    'cello',
    'cell phone',
    'chair',
    'chandelier',
    'church',
    'circle',
    'clarinet',
    'clock',
    'cloud',
    'coffee cup',
    'compass',
    'computer',
    'cookie',
    'cooler',
    'couch',
    'cow',
    'crab',
    'crayon',
    'crocodile',
    'crown',
    'cruise ship',
    'cup',
    'diamond',
    'dishwasher',
    'diving board',
    'dog',
    'dolphin',
    'donut',
    'door',
    'dragon',
    'dresser',
    'drill',
    'drums',
    'duck',
    'dumbbell',
    'ear',
    'elbow',
    'elephant',
    'envelope',
    'eraser',
    'eye',
    'eyeglasses',
    'face',
    'fan',
    'feather',
    'fence',
    'finger',
    'fire hydrant',
    'fireplace',
    'firetruck',
    'fish',
    'flamingo',
    'flashlight',
    'flip flops',
    'floor lamp',
    'flower',
    'flying saucer',
    'foot',
    'fork',
    'frog',
    'frying pan',
    'garden',
    'garden hose',
    'giraffe',
    'goatee',
    'golf club',
    'grapes',
    'grass',
    'guitar',
    'hamburger',
    'hammer',
    'hand',
    'harp',
    'hat',
    'headphones',
    'hedgehog',
    'helicopter',
    'helmet',
    'hexagon',
    'hockey puck',
    'hockey stick',
    'horse',
    'hospital',
    'hot air balloon',
    'hot dog',
    'hot tub',
    'hourglass',
    'house',
    'house plant',
    'hurricane',
    'ice cream',
    'jacket',
    'jail',
    'kangaroo',
    'key',
    'keyboard',
    'knee',
    'knife',
    'ladder',
    'lantern',
    'laptop',
    'leaf',
    'leg',
    'light bulb',
    'lighter',
    'lighthouse',
    'lightning',
    'line',
    'lion',
    'lipstick',
    'lobster',
    'lollipop',
    'mailbox',
    'map',
    'marker',
    'matches',
    'megaphone',
    'mermaid',
    'microphone',
    'microwave',
    'monkey',
    'moon',
    'mosquito',
    'motorbike',
    'mountain',
    'mouse',
    'moustache',
    'mouth',
    'mug',
    'mushroom',
    'nail',
    'necklace',
    'nose',
    'ocean',
    'octagon',
    'octopus',
    'onion',
    'oven',
    'owl',
    'paintbrush',
    'paint can',
    'palm tree',
    'panda',
    'pants',
    'paper clip',
    'parachute',
    'parrot',
    'passport',
    'peanut',
    'pear',
    'peas',
    'pencil',
    'penguin',
    'piano',
    'pickup truck',
    'picture frame',
    'pig',
    'pillow',
    'pineapple',
    'pizza',
    'pliers',
    'police car',
    'pond',
    'pool',
    'popsicle',
    'postcard',
    'potato',
    'power outlet',
    'purse',
    'rabbit',
    'raccoon',
    'radio',
    'rain',
    'rainbow',
    'rake',
    'remote control',
    'rhinoceros',
    'rifle',
    'river',
    'roller coaster',
    'rollerskates',
    'sailboat',
    'sandwich',
    'saw',
    'saxophone',
    'school bus',
    'scissors',
    'scorpion',
    'screwdriver',
    'sea turtle',
    'see saw',
    'shark',
    'sheep',
    'shoe',
    'shorts',
    'shovel',
    'sink',
    'skateboard',
    'skull',
    'skyscraper',
    'sleeping bag',
    'smiley face',
    'snail',
    'snake',
    'snorkel',
    'snowflake',
    'snowman',
    'soccer ball',
    'sock',
    'speedboat',
    'spider',
    'spoon',
    'spreadsheet',
    'square',
    'squiggle',
    'squirrel',
    'stairs',
    'star',
    'steak',
    'stereo',
    'stethoscope',
    'stitches',
    'stop sign',
    'stove',
    'strawberry',
    'streetlight',
    'string bean',
    'submarine',
    'suitcase',
    'sun',
    'swan',
    'sweater',
    'swing set',
    'sword',
    'syringe',
    'table',
    'teapot',
    'teddy-bear',
    'telephone',
    'television',
    'tennis racquet',
    'tent',
    'The Eiffel Tower',
    'The Great Wall of China',
    'The Mona Lisa',
    'tiger',
    'toaster',
    'toe',
    'toilet',
    'tooth',
    'toothbrush',
    'toothpaste',
    'tornado',
    'tractor',
    'traffic light',
    'train',
    'tree',
    'triangle',
    'trombone',
    'truck',
    'trumpet',
    't-shirt',
    'umbrella',
    'underwear',
    'van',
    'vase',
    'violin',
    'washing machine',
    'watermelon',
    'waterslide',
    'whale',
    'wheel',
    'windmill',
    'wine bottle',
    'wine glass',
    'wristwatch',
    'yoga',
    'zebra',
    'zigzag']
        return classes[index]
    elif language == 'ja':
        classes = [ '航空母艦',
 '飛行機',
 '目覚まし時計',
 '救急車',
 '天使',
 '鳥',
 'アリ',
 'アンヴィル',
 'リンゴ',
 'アーム',
 'アスパラガス',
 '斧',
 'リュックサック',
 'バナナ',
 '包帯',
 'バーン',
 '野球',
 'バット',
 'バスケット',
 'バスケットボール',
 'バット',
 'バスタブ',
 'ビーチ',
 '熊',
 'ヒゲ',
 'ベッド',
 'ハチ',
 'ベルト',
 'ベンチ',
 '自転車',
 '双眼鏡',
 '鳥',
 'バースデーケーキ',
 'ブラックベリー',
 'ブルーベリー',
 '本',
 'ブーメラン',
 'ボトルキャップ',
 '蝶ネクタイ',
 'ブレスレット',
 '脳',
 'パン',
 '橋',
 'ブロッコリー',
 'ブルーム',
 'バケツ',
 'ブルドーザー',
 'バス',
 'ブッシュ',
 '蝶',
 'カクタス',
 'ケーキ',
 '電卓',
 'カレンダー',
 'キャメル',
 'カメラ',
 'カモフラージュ',
 '焚き火',
 'キャンドル',
 'キャノン',
 'カヌー',
 '車',
 'キャロット',
 '城',
 '猫',
 '扇風機',
 'セロ',
 '携帯電話',
 '椅子',
 'シャンデリア',
 '教会',
 'サークル',
 'クラリネット',
 '時計',
 '雲',
 'コーヒーカップ',
 'コンパス',
 'コンピュータ',
 'クッキー',
 'クーラー',
 'ソファー',
 '牛',
 'カニ',
 'クレヨン',
 'クロコダイル',
 '王冠',
 '船',
 'カップ',
 'ダイヤモンド',
 '食器洗い機',
 '飛び込み台',
 '犬',
 'ドルフィン',
 'ドーナツ',
 'ドア',
 'ドラゴン',
 'ドレッサー',
 'ドリル',
 'ドラム',
 'ダック',
 'ダンベル',
 '耳',
 'エルボー',
 'エレファント',
 '封筒',
 '消しゴム',
 '目',
 '眼鏡',
 '顔',
 'ファン',
 '羽',
 'フェンス',
 '指',
 '消火栓',
 '暖炉',
 '消防車',
 '魚',
 'フラミンゴ',
 '懐中電灯',
 'ビーチサンダル',
 'ランプ',
 'フラワー',
 'ユーフォー',
 'フット',
 'フォーク',
 'カエル',
 'フライパン',
 'ガーデン',
 'ホース',
 'ジラフ',
 'あごひげ',
 'ゴルフクラブ',
 'ブドウ',
 '草',
 'ギター',
 'ハンバーガー',
 'ハンマー',
 '手',
 'ハープ',
 '帽子',
 'ヘッドフォン',
 'ヘッジホッグ',
 'ヘリコプター',
 'ヘルメット',
 '六角形',
 'アイスホッケー',
 'アイスホッケー',
 '馬',
 '病院',
 '熱気球',
 'ホットドッグ',
 '風呂',
 '砂時計',
 '家',
 '観葉植物',
 'ハリケーン',
 'アイスクリーム',
 'ジャケット',
 '刑務所',
 'カンガルー',
 'キー',
 'キーボード',
 '膝',
 'ナイフ',
 'ハシゴ',
 'ランタン',
 'ノートパソコン',
 '葉',
 '脚',
 '電球',
 'ライター',
 '灯台',
 'ライトニング',
 'ライン',
 'ライオン',
 'リップスティック',
 'ロブスター',
 'ロリポップ',
 'メールボックス',
 '地図',
 'マーカー',
 'マッチ',
 'メガホン',
 'マーメイド',
 'マイク',
 'マイクロウェーブ',
 'モンキー',
 '月',
 'モスキート',
 'バイク',
 '山',
 'マウス',
 '口ひげ',
 '口',
 'マグカップ',
 'キノコ',
 'ネイル',
 'ネックレス',
 '鼻',
 'オーシャン',
 'オクタゴン',
 'タコ',
 '玉ねぎ',
 'オーブン',
 'フクロウ',
 '絵筆',
 'ペンキ',
 '木',
 'パンダ',
 'パンツ',
 'クリップ',
 'パラシュート',
 'オウム',
 'パスポート',
 'ピーナッツ',
 '梨',
 '豆',
 '鉛筆',
 'ペンギン',
 'ピアノ',
 'ピックアップトラック',
 '額縁',
 'ブタ',
 '枕',
 'パイナップル',
 'ピザ',
 'ペンチ',
 'パトカー',
 '池',
 'プール',
 'アイスキャンディー',
 'ポストカード',
 'ポテト',
 'コンセント',
 '財布',
 'ウサギ',
 'アライグマ',
 'ラジオ',
 '雨',
 'レインボー',
 'レーキ',
 'リモコン',
 'サイ',
 'ライフル',
 'リバー',
 'ローラーコースター',
 'ローラースケート',
 '帆船',
 'サンドイッチ',
 'のこぎり',
 'サクソフォーン',
 'スクールバス',
 'はさみ',
 'サソリ',
 'ドライバー',
 'ウミガメ',
 'シーソー',
 'サメ',
 '羊',
 '靴',
 'ショートパンツ',
 'ショベル',
 'シンク',
 'スケートボード',
 '頭蓋骨',
 '高層ビル',
 '寝袋',
 'スマイリーフェイス',
 'カタツムリ',
 'ヘビ',
 'シュノーケル',
 '雪',
 '雪だるま',
 'サッカーボール',
 '靴下',
 'ボート',
 'クモ',
 'スプーン',
 'スプレッドシート',
 '正方形',
 'スクイッグル',
 'リス',
 '階段',
 '星',
 'ステーキ',
 'ステレオ',
 '聴診器',
 'ステッチ',
 'ストップサイン',
 'ストーブ',
 'ストロベリー',
 '街路灯',
 'インゲン豆',
 '潜水艦',
 'スーツケース',
 '太陽',
 '白鳥',
 'セーター',
 'スイングセット',
 '剣',
 '注射器',
 'テーブル',
 'ティーポット',
 'テディベア',
 '電話',
 'テレビ',
 'テニスラケット',
 'テント',
 'エッフェル塔',
 '万里の長城',
 'モナリザ',
 'トラ',
 'トースター',
 'つま先',
 'トイレ',
 '歯',
 '歯ブラシ',
 '歯磨き粉',
 'トルネード',
 'トラクター',
 '信号機',
 '電車',
 '木',
 'トライアングル',
 'トロンボーン',
 'トラック',
 'トランペット',
 'Tシャツ',
 '傘',
 '下着',
 'バン',
 '花瓶',
 'バイオリン',
 '洗濯機',
 'スイカ',
 'ウォータースライド',
 'クジラ',
 'ホイール',
 '風車',
 'ワインボトル',
 'ワイングラス',
 '腕時計',
 'ヨガ',
 'ゼブラ',
 'ジグザグ']

        return classes[index]
    else:
        raise ValueError('Invalid language. Language must be either en or ja.')