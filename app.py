from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode, urlunparse
from PIL import Image, ImageOps, ImageEnhance
import requests, io, os, time, threading, json, re, base64, math

app = Flask(__name__)
CORS(app)

SHOP_BASE_URL = os.environ.get('SHOP_BASE_URL', 'https://freeorder1.cafe24.com').rstrip('/')
DATA_DIR = os.environ.get('DATA_DIR', '/var/data')
INDEX_FILE = os.path.join(DATA_DIR, 'semantic_product_index.json')
ANCHOR_FILE = os.path.join(DATA_DIR, 'fashion_anchor_embeddings.json')
MAX_CATEGORY_PAGES = int(os.environ.get('MAX_CATEGORY_PAGES', '200'))
REQUEST_TIMEOUT = int(os.environ.get('REQUEST_TIMEOUT', '20'))
RESULT_LIMIT = int(os.environ.get('RESULT_LIMIT', '12'))
MAX_SEARCH_IMAGES = int(os.environ.get('MAX_SEARCH_IMAGES', '4'))
MAX_QUERY_VIEWS = int(os.environ.get('MAX_QUERY_VIEWS', '2'))
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN', '').strip()
REPLICATE_MODEL = os.environ.get('REPLICATE_MODEL', 'openai/clip').strip()
REPLICATE_API = 'https://api.replicate.com/v1'
INDEX_VERSION = 7
SEARCH_MODE = 'semantic_fashion_clip_design_anchors_v7'

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/150.0 Safari/537.36'
session = requests.Session()
session.headers.update({'User-Agent': USER_AGENT, 'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8'})

product_index = []
indexing = False
last_indexed_at = None
index_lock = threading.Lock()
anchor_embeddings = {}
reindex_progress = {
    'running': False, 'force': False, 'started_at': None, 'finished_at': None,
    'current': 0, 'total': 0, 'added': 0, 'reused': 0, 'failed': 0,
    'ai_calls': 0, 'message': '대기 중', 'error': None,
}

FASHION_ANCHORS = [
    'a horizontal striped knit cardigan',
    'a vertical striped knit top',
    'a cable knit sweater with vertical braids',
    'a ribbed knit sweater',
    'a plain solid knit sweater',
    'a cardigan with buttons all the way down the front',
    'a pullover sweater with a short henley button placket',
    'a crew neck sweater',
    'a henley neck sweater',
    'a v neck sweater',
    'a collared knit shirt',
    'a zip up knit cardigan',
    'a cropped knit top',
    'an oversized relaxed fit sweater',
    'a slim fitted knit top',
    'a long sleeve knit sweater',
    'a short sleeve knit top',
    'a raglan sleeve sweater',
    'a drop shoulder sweater',
    'a textured knit sweater',
    'a smooth fine gauge knit sweater',
    'a waffle knit top',
    'a mesh knit sweater',
    'a sweater with contrast stitching',
    'a knit top with chest pocket',
    'a button front cardigan',
    'a pullover knit sweater',
]


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def get_product_no(url):
    if not url:
        return None
    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query))
    if params.get('product_no'):
        return str(params['product_no'])
    parts = [p for p in parsed.path.split('/') if p]
    if 'product' in parts:
        try:
            remain = parts[parts.index('product') + 1:]
            if len(remain) >= 2 and remain[1].isdigit():
                return remain[1]
        except Exception:
            pass
    m = re.search(r'/product/(?:[^/]+/)?(\d+)(?:/|$)', parsed.path)
    return m.group(1) if m else None


def save_index():
    ensure_data_dir()
    payload = {
        'version': INDEX_VERSION, 'search_mode': SEARCH_MODE, 'model': REPLICATE_MODEL,
        'shop': SHOP_BASE_URL, 'last_indexed_at': last_indexed_at,
        'count': len(product_index), 'products': product_index,
    }
    temp = INDEX_FILE + '.tmp'
    with open(temp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))
    os.replace(temp, INDEX_FILE)


def load_index():
    global product_index, last_indexed_at
    ensure_data_dir()
    if not os.path.exists(INDEX_FILE):
        return False
    try:
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if payload.get('version') != INDEX_VERSION or payload.get('search_mode') != SEARCH_MODE or payload.get('model') != REPLICATE_MODEL:
            return False
        products = payload.get('products', [])
        if not isinstance(products, list):
            return False
        product_index = products
        last_indexed_at = payload.get('last_indexed_at')
        print(f'[INDEX LOAD] {len(product_index)} products', flush=True)
        return True
    except Exception as e:
        print('[INDEX LOAD ERROR]', repr(e), flush=True)
        return False


def load_anchor_cache():
    global anchor_embeddings
    ensure_data_dir()
    if not os.path.exists(ANCHOR_FILE):
        return False
    try:
        with open(ANCHOR_FILE, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        data = payload.get('anchors', {})
        if payload.get('model') != REPLICATE_MODEL or not all(x in data for x in FASHION_ANCHORS):
            return False
        anchor_embeddings = data
        return True
    except Exception:
        return False


def save_anchor_cache():
    ensure_data_dir()
    temp = ANCHOR_FILE + '.tmp'
    with open(temp, 'w', encoding='utf-8') as f:
        json.dump({'model': REPLICATE_MODEL, 'anchors': anchor_embeddings}, f, ensure_ascii=False, separators=(',', ':'))
    os.replace(temp, ANCHOR_FILE)


def normalize_url(url, allow_external=False):
    if not url:
        return None
    absolute = urljoin(SHOP_BASE_URL + '/', url)
    parsed = urlparse(absolute)
    if not allow_external:
        base_host = urlparse(SHOP_BASE_URL).netloc
        if parsed.netloc and parsed.netloc != base_host:
            return None
    return urlunparse(parsed._replace(fragment=''))


def is_product_url(url):
    if not url:
        return False
    path = urlparse(url).path.lower()
    blocked = ['/board/', '/product/image_zoom', '/product/zoom', '/product/list.html', '/product/search.html', '/order/', '/myshop/', '/member/']
    return not any(x in path for x in blocked) and get_product_no(url) is not None


def is_category_url(url):
    if not url:
        return False
    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query))
    return '/category/' in parsed.path.lower() or (parsed.path.lower().endswith('/product/list.html') and 'cate_no' in params)


def with_page(url, page_no):
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    query['page'] = str(page_no)
    return urlunparse(parsed._replace(query=urlencode(query)))


def fetch_html(url):
    r = session.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or r.encoding
    return r.text


def discover_categories():
    soup = BeautifulSoup(fetch_html(SHOP_BASE_URL + '/'), 'html.parser')
    found = set()
    for a in soup.find_all('a', href=True):
        url = normalize_url(a.get('href'))
        if url and is_category_url(url):
            found.add(url)
    return sorted(found)


def discover_product_urls():
    by_no = {}
    def add(url):
        if url and is_product_url(url):
            pno = get_product_no(url)
            if pno:
                by_no.setdefault(str(pno), url)
    try:
        soup = BeautifulSoup(fetch_html(SHOP_BASE_URL + '/'), 'html.parser')
        for a in soup.find_all('a', href=True):
            add(normalize_url(a.get('href')))
    except Exception as e:
        print('[DISCOVER MAIN]', repr(e), flush=True)
    try:
        categories = discover_categories()
    except Exception:
        categories = []
    for category_url in categories:
        previous = None
        for page in range(1, MAX_CATEGORY_PAGES + 1):
            try:
                soup = BeautifulSoup(fetch_html(with_page(category_url, page)), 'html.parser')
            except Exception:
                break
            nums = set()
            for a in soup.find_all('a', href=True):
                url = normalize_url(a.get('href'))
                if url and is_product_url(url):
                    pno = get_product_no(url)
                    if pno:
                        nums.add(str(pno)); add(url)
            if not nums:
                break
            signature = tuple(sorted(nums))
            if signature == previous:
                break
            previous = signature
    return [by_no[k] for k in sorted(by_no, key=lambda x: int(x) if x.isdigit() else x)]


def find_canonical_product_url(soup, fallback):
    candidates = []
    canonical = soup.find('link', attrs={'rel': 'canonical'})
    if canonical and canonical.get('href'):
        candidates.append(canonical.get('href'))
    og = soup.find('meta', attrs={'property': 'og:url'})
    if og and og.get('content'):
        candidates.append(og.get('content'))
    candidates.append(fallback)
    for c in candidates:
        url = normalize_url(c)
        if url and is_product_url(url):
            return url
    return fallback


def clean_price(value):
    text = '' if value is None else str(value).strip()
    nums = re.sub(r'[^0-9]', '', text)
    try:
        return f'{int(nums):,}원' if nums else text
    except Exception:
        return text


def extract_price(soup):
    for attr, value in [('property','product:price:amount'),('property','og:price:amount'),('name','product:price:amount'),('name','price')]:
        tag = soup.find('meta', attrs={attr: value})
        if tag and tag.get('content'):
            p = clean_price(tag.get('content'))
            if p: return p
    for selector in ['.xans-product-detail .price','.xans-product-detaildesign td span','.product_price','.price','[data-price]']:
        el = soup.select_one(selector)
        if el:
            p = clean_price(el.get('data-price') or el.get_text(' ', strip=True))
            if p: return p
    return ''


def looks_like_product_image(url):
    if not url: return False
    low = url.lower()
    return not any(x in low for x in ['icon','logo','banner','btn_','button','loading','common','layout','board','member','review','coupon','soldout','wish','basket'])


def extract_product_info(product_url):
    soup = BeautifulSoup(fetch_html(product_url), 'html.parser')
    canonical = find_canonical_product_url(soup, product_url)
    pno = get_product_no(canonical) or get_product_no(product_url)
    title = ''
    og_title = soup.find('meta', attrs={'property': 'og:title'})
    if og_title and og_title.get('content'):
        title = og_title.get('content').strip()
    if not title and soup.title:
        title = soup.title.get_text(' ', strip=True)

    def abs_img(src):
        if not src: return None
        u = urljoin(canonical, src)
        return u if looks_like_product_image(u) else None

    display = None
    og_img = soup.find('meta', attrs={'property': 'og:image'})
    if og_img and og_img.get('content'):
        display = abs_img(og_img.get('content'))
    if not display:
        for selector in ['.keyImg img','.thumbnail img','.prdImg img','img.BigImage']:
            img = soup.select_one(selector)
            if img:
                display = abs_img(img.get('ec-data-src') or img.get('data-src') or img.get('src'))
                if display: break

    images = []
    def add(src):
        u = abs_img(src)
        if u and u not in images:
            images.append(u)
    if display: images.append(display)
    for selector in ['#prdDetail img','.xans-product-additional img','.detailArea img','.product-detail img','.prdDetail img','.xans-product-addimage img']:
        for img in soup.select(selector):
            add(img.get('ec-data-src') or img.get('data-src') or img.get('data-original') or img.get('src'))
            if len(images) >= MAX_SEARCH_IMAGES: break
        if len(images) >= MAX_SEARCH_IMAGES: break

    return {'product_no': str(pno or ''), 'product_url': canonical, 'title': title, 'image_url': display, 'image_urls': images[:MAX_SEARCH_IMAGES], 'price': extract_price(soup)}


def download_image(url):
    r = session.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return ImageOps.exif_transpose(Image.open(io.BytesIO(r.content))).convert('RGB')


def resize_for_ai(image, max_side=768):
    image = ImageOps.exif_transpose(image).convert('RGB')
    if max(image.size) > max_side:
        scale = max_side / float(max(image.size))
        image = image.resize((max(1,int(image.width*scale)), max(1,int(image.height*scale))), Image.Resampling.LANCZOS)
    return image


def design_grayscale(image):
    image = resize_for_ai(image)
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.05)
    return gray.convert('RGB')


def make_query_views(image):
    image = ImageOps.exif_transpose(image).convert('RGB')
    w,h = image.size
    views = []
    def crop(l,t,r,b):
        box=(int(w*l),int(h*t),int(w*r),int(h*b))
        if box[2]-box[0] >= 120 and box[3]-box[1] >= 120:
            views.append(image.crop(box))
    crop(0.05,0.00,0.95,0.72)
    crop(0.14,0.00,0.86,0.58)
    return (views or [image])[:MAX_QUERY_VIEWS]


def image_to_data_uri(image):
    buf = io.BytesIO()
    resize_for_ai(image).save(buf, format='JPEG', quality=90, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def require_token():
    if not REPLICATE_API_TOKEN:
        raise RuntimeError('REPLICATE_API_TOKEN 환경변수가 없습니다. Render Environment에 Replicate API Token을 추가해주세요.')


def replicate_prediction(input_payload):
    require_token()
    parts = REPLICATE_MODEL.split('/',1)
    if len(parts)!=2:
        raise RuntimeError('REPLICATE_MODEL은 owner/model 형식이어야 합니다.')
    owner, model = parts
    url = f'{REPLICATE_API}/models/{owner}/{model}/predictions'
    headers = {'Authorization': f'Bearer {REPLICATE_API_TOKEN}', 'Content-Type':'application/json', 'Prefer':'wait=60'}
    r = requests.post(url, headers=headers, json={'input': input_payload}, timeout=75)
    if r.status_code >= 400:
        raise RuntimeError(f'Replicate API 오류 {r.status_code}: {r.text[:300]}')
    data = r.json()
    if data.get('status') == 'succeeded':
        return data
    get_url = (data.get('urls') or {}).get('get')
    if not get_url: return data
    deadline = time.time()+90
    while time.time() < deadline:
        time.sleep(1)
        rr = requests.get(get_url, headers={'Authorization': f'Bearer {REPLICATE_API_TOKEN}'}, timeout=20)
        rr.raise_for_status(); cur = rr.json(); status = cur.get('status')
        if status == 'succeeded': return cur
        if status in ('failed','canceled'):
            raise RuntimeError('Replicate prediction 실패: '+str(cur.get('error') or status))
    raise RuntimeError('Replicate AI 응답 시간이 초과되었습니다.')


def output_embedding(data):
    out = data.get('output')
    if isinstance(out, dict) and isinstance(out.get('embedding'), list):
        return [float(x) for x in out['embedding']]
    if isinstance(out, list) and out and all(isinstance(x,(int,float)) for x in out):
        return [float(x) for x in out]
    raise RuntimeError('CLIP embedding 응답 형식을 읽지 못했습니다.')


def normalize_vector(v):
    n = math.sqrt(sum(float(x)*float(x) for x in v))
    return [float(x)/n for x in v] if n > 1e-12 else [float(x) for x in v]


def cosine(a,b):
    if not a or not b or len(a)!=len(b): return 0.0
    return sum(float(x)*float(y) for x,y in zip(a,b))


def clip_image_embedding(image):
    processed = design_grayscale(image)
    return normalize_vector(output_embedding(replicate_prediction({'image': image_to_data_uri(processed)})))


def clip_text_embedding(text):
    return normalize_vector(output_embedding(replicate_prediction({'text': text})))


def ensure_anchor_embeddings():
    global anchor_embeddings
    if anchor_embeddings and all(x in anchor_embeddings for x in FASHION_ANCHORS): return
    if load_anchor_cache(): return
    require_token(); anchor_embeddings = {}
    for i,text in enumerate(FASHION_ANCHORS,1):
        reindex_progress['message'] = f'디자인 기준 AI 준비 {i}/{len(FASHION_ANCHORS)}'
        anchor_embeddings[text] = clip_text_embedding(text)
        reindex_progress['ai_calls'] += 1
    save_anchor_cache()


def attribute_signature(embeddings):
    ensure_anchor_embeddings()
    return [max((cosine(e, anchor_embeddings[a]) for e in embeddings), default=0.0) for a in FASHION_ANCHORS]


def centered_cosine(a,b):
    if not a or not b or len(a)!=len(b): return 0.0
    ma=sum(a)/len(a); mb=sum(b)/len(b)
    aa=[x-ma for x in a]; bb=[x-mb for x in b]
    na=math.sqrt(sum(x*x for x in aa)); nb=math.sqrt(sum(x*x for x in bb))
    return sum(x*y for x,y in zip(aa,bb))/(na*nb) if na>1e-12 and nb>1e-12 else 0.0


def top_anchor_labels(sig, top_k=6):
    pairs=sorted(zip(FASHION_ANCHORS,sig), key=lambda x:x[1], reverse=True)
    return [name for name,_ in pairs[:top_k]]


def anchor_overlap(q,p):
    qa=set(top_anchor_labels(q,6)); pa=set(top_anchor_labels(p,6)); u=qa|pa
    return len(qa&pa)/len(u) if u else 0.0


def semantic_score(query_embeddings, query_sig, product):
    p_embs=product.get('embeddings') or []
    if not p_embs: return 0,0,0,0
    image_sim=max((cosine(q,p) for q in query_embeddings for p in p_embs), default=0.0)
    p_sig=product.get('attribute_signature') or attribute_signature(p_embs)
    attr=centered_cosine(query_sig,p_sig)
    overlap=anchor_overlap(query_sig,p_sig)
    final=image_sim*0.60 + attr*0.30 + overlap*0.10
    return final,image_sim,attr,overlap


def usable_existing(item):
    return isinstance(item,dict) and isinstance(item.get('embeddings'),list) and item.get('embeddings') and isinstance(item.get('attribute_signature'),list)


def build_index(force=False):
    global product_index,indexing,last_indexed_at
    with index_lock:
        if indexing: return {'success':False,'message':'이미 인덱싱 중입니다.'}
        indexing=True
    reindex_progress.update({'running':True,'force':bool(force),'started_at':time.strftime('%Y-%m-%d %H:%M:%S'),'finished_at':None,'current':0,'total':0,'added':0,'reused':0,'failed':0,'ai_calls':0,'message':'AI 연결 확인 중','error':None})
    try:
        require_token(); ensure_anchor_embeddings()
        existing={str(x.get('product_no')):x for x in product_index if x.get('product_no')}
        reindex_progress['message']='상품 URL 수집 중'
        urls=discover_product_urls(); reindex_progress['total']=len(urls)
        new=[]; added=reused=failed=0
        for idx,url in enumerate(urls,1):
            reindex_progress['current']=idx; reindex_progress['message']=f'상품 디자인 AI 분석 중 {idx}/{len(urls)}'
            pno=get_product_no(url); old=existing.get(str(pno))
            if not force and usable_existing(old):
                new.append(old); reused+=1; reindex_progress['reused']=reused; continue
            try:
                info=extract_product_info(url); embs=[]; used=[]
                for image_url in info.get('image_urls',[]):
                    try:
                        image=download_image(image_url)
                        if image.width<180 or image.height<180: continue
                        embs.append(clip_image_embedding(image)); used.append(image_url); reindex_progress['ai_calls']+=1
                    except Exception as ie:
                        print('[EMBED IMAGE ERROR]',image_url,repr(ie),flush=True)
                if not embs:
                    failed+=1; reindex_progress['failed']=failed; continue
                sig=attribute_signature(embs)
                new.append({'product_no':str(info['product_no']),'title':str(info.get('title','')),'product_url':str(info.get('product_url','')),'image_url':str(info.get('image_url','') or ''),'price':str(info.get('price','')),'search_image_urls':used,'embeddings':embs,'attribute_signature':sig,'top_design_features':top_anchor_labels(sig,6)})
                added+=1; reindex_progress['added']=added
            except Exception as e:
                failed+=1; reindex_progress['failed']=failed; print('[INDEX PRODUCT ERROR]',url,repr(e),flush=True)
        dedup={str(x.get('product_no')):x for x in new if x.get('product_no')}
        product_index=list(dedup.values()); last_indexed_at=time.strftime('%Y-%m-%d %H:%M:%S'); save_index()
        reindex_progress.update({'running':False,'finished_at':time.strftime('%Y-%m-%d %H:%M:%S'),'current':len(urls),'total':len(urls),'added':added,'reused':reused,'failed':failed,'message':'완료','error':None})
        return {'success':True,'indexed':len(product_index),'added':added,'reused':reused,'failed':failed,'ai_calls':reindex_progress['ai_calls'],'last_indexed_at':last_indexed_at}
    except Exception as e:
        reindex_progress.update({'running':False,'finished_at':time.strftime('%Y-%m-%d %H:%M:%S'),'message':'오류 발생','error':str(e)})
        print('[INDEX ERROR]',repr(e),flush=True); return {'success':False,'error':str(e)}
    finally:
        indexing=False


def run_reindex_background(force=False):
    build_index(force=force)


@app.route('/',methods=['GET'])
def home():
    return jsonify({'success':True,'message':'Semantic Fashion Image Search API is running.','shop':SHOP_BASE_URL,'search_mode':SEARCH_MODE,'model':REPLICATE_MODEL,'ai_token_configured':bool(REPLICATE_API_TOKEN),'indexed_products':len(product_index),'indexing':indexing,'last_indexed_at':last_indexed_at})


@app.route('/status',methods=['GET'])
def status():
    return jsonify({'success':True,'shop':SHOP_BASE_URL,'search_mode':SEARCH_MODE,'model':REPLICATE_MODEL,'ai_token_configured':bool(REPLICATE_API_TOKEN),'indexed_products':len(product_index),'indexing':indexing,'last_indexed_at':last_indexed_at,'progress':reindex_progress})


@app.route('/reindex/start',methods=['POST'])
def reindex_start():
    force=request.args.get('force','0')=='1'
    if indexing or reindex_progress.get('running'):
        return jsonify({'success':False,'message':'이미 재색인 중입니다.','progress':reindex_progress}),409
    if not REPLICATE_API_TOKEN:
        return jsonify({'success':False,'message':'REPLICATE_API_TOKEN이 설정되지 않았습니다.'}),400
    threading.Thread(target=run_reindex_background,kwargs={'force':force},daemon=True).start()
    return jsonify({'success':True,'message':'AI 디자인 재색인을 시작했습니다.','force':force})


@app.route('/reindex/progress',methods=['GET'])
def reindex_progress_api():
    return jsonify({'success':True,'progress':reindex_progress,'indexed_products':len(product_index),'last_indexed_at':last_indexed_at})


@app.route('/reindex',methods=['POST'])
def reindex_compat():
    return reindex_start()


@app.route('/admin/reindex',methods=['GET'])
def admin_reindex_page():
    token_state='설정됨' if REPLICATE_API_TOKEN else '설정 안 됨'
    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI 상품 이미지 검색 DB 관리</title><style>body{{font-family:Arial,sans-serif;background:#f5f5f5;margin:0;padding:40px 20px;color:#111}}.box{{max-width:720px;margin:auto;background:#fff;border-radius:18px;padding:32px;box-shadow:0 8px 30px rgba(0,0,0,.08)}}h1{{font-size:24px;margin:0 0 12px}}.desc{{color:#666;line-height:1.7;margin-bottom:20px}}.info{{background:#f7f7f7;border-radius:12px;padding:14px;margin-bottom:18px;font-size:14px;line-height:1.6}}button{{width:100%;border:0;border-radius:12px;padding:16px;font-size:16px;font-weight:700;cursor:pointer;margin-top:10px}}.refresh{{background:#111;color:#fff}}.force{{background:#eee;color:#111}}button:disabled{{opacity:.45;cursor:not-allowed}}.barwrap{{height:12px;background:#ececec;border-radius:999px;overflow:hidden;margin-top:18px}}.bar{{height:100%;width:0;background:#111;transition:width .3s}}#status{{font-weight:700;margin-top:12px}}#result{{white-space:pre-wrap;background:#f7f7f7;padding:16px;border-radius:12px;margin-top:16px;line-height:1.55}}</style></head><body><div class="box"><h1>AI 상품 이미지 검색 DB 관리</h1><div class="desc">고객 착용샷을 <b>상품 디자인/디테일 의미</b>로 분석해 누끼컷·디테일컷과 비교합니다. 색상 영향은 낮추고, 스트라이프·케이블·버튼 구조·넥라인 같은 특징을 함께 사용합니다.</div><div class="info">AI 모델: {REPLICATE_MODEL}<br>REPLICATE_API_TOKEN: <b>{token_state}</b><br>검색 방식: Semantic CLIP + Fashion Design Anchors</div><button id="refreshBtn" class="refresh" onclick="startJob(false)">신규 상품 AI 반영</button><button id="forceBtn" class="force" onclick="startJob(true)">전체 AI 강제 재생성</button><div class="barwrap"><div id="bar" class="bar"></div></div><div id="status">상태 확인 중...</div><div id="result">대기 중</div></div><script>let timer=null;function buttons(v){{document.getElementById('refreshBtn').disabled=v;document.getElementById('forceBtn').disabled=v;}}async function startJob(force){{if(force&&!confirm('모든 상품을 외부 AI로 다시 분석합니다. API 사용량이 발생할 수 있습니다. 계속할까요?'))return;buttons(true);try{{const r=await fetch('/reindex/start'+(force?'?force=1':''),{{method:'POST'}});const d=await r.json();document.getElementById('result').textContent=JSON.stringify(d,null,2);if(!r.ok){{buttons(false);return;}}poll();}}catch(e){{document.getElementById('result').textContent='오류: '+e.message;buttons(false);}}}}async function poll(){{try{{const r=await fetch('/reindex/progress',{{cache:'no-store'}});const d=await r.json();const p=d.progress||{{}};const total=Number(p.total||0),current=Number(p.current||0);const pct=total?Math.round(current/total*100):0;document.getElementById('bar').style.width=pct+'%';document.getElementById('status').textContent=(p.running?'진행 중':(p.message||'대기'))+(total?` · ${{current}}/${{total}} (${{pct}}%)`:'');document.getElementById('result').textContent=`상태: ${{p.message||'-'}}\n현재: ${{current}}/${{total||'-'}}\n추가: ${{p.added||0}}\n재사용: ${{p.reused||0}}\n실패: ${{p.failed||0}}\nAI 호출: ${{p.ai_calls||0}}\n오류: ${{p.error||'없음'}}\n인덱스 상품: ${{d.indexed_products||0}}\n마지막 완료: ${{d.last_indexed_at||'-'}}`;buttons(Boolean(p.running));if(p.running){{clearTimeout(timer);timer=setTimeout(poll,1500);}}}}catch(e){{document.getElementById('result').textContent='진행 상태 오류: '+e.message;buttons(false);}}}}poll();</script></body></html>'''


@app.route('/search',methods=['POST'])
def image_search():
    try:
        if not REPLICATE_API_TOKEN:
            return jsonify({'error':'AI 검색 설정이 완료되지 않았습니다.','detail':'REPLICATE_API_TOKEN 환경변수를 설정해주세요.'}),503
        if 'image' not in request.files:
            return jsonify({'error':'이미지 파일이 전송되지 않았습니다.'}),400
        if not product_index: load_index()
        if not product_index:
            return jsonify({'error':'상품 AI 검색 DB가 비어 있습니다.','detail':'/admin/reindex에서 전체 AI 강제 재생성을 먼저 실행해주세요.'}),503
        raw=request.files['image'].read()
        if not raw: return jsonify({'error':'업로드 이미지가 비어 있습니다.'}),400
        query=ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
        q_embs=[]
        for view in make_query_views(query):
            try: q_embs.append(clip_image_embedding(view))
            except Exception as e: print('[QUERY EMBED ERROR]',repr(e),flush=True)
        if not q_embs: return jsonify({'error':'착용샷 AI 분석에 실패했습니다.'}),500
        q_sig=attribute_signature(q_embs); q_features=top_anchor_labels(q_sig,6); matches=[]
        for product in product_index:
            try:
                final,image_sim,attr_sim,overlap=semantic_score(q_embs,q_sig,product)
                percent=max(0.0,min(99.0,final*100.0))
                matches.append({'score':round(final,6),'similarity_percent':round(percent,1),'design_similarity_percent':round(percent,1),'semantic_similarity':round(image_sim,6),'design_attribute_similarity':round(attr_sim,6),'design_anchor_overlap':round(overlap,6),'title':str(product.get('title','')),'product_no':str(product.get('product_no','')),'product_url':str(product.get('product_url','')),'image_url':str(product.get('image_url','')),'price':str(product.get('price','')),'matched_design_features':product.get('top_design_features',[])})
            except Exception as e: print('[SEARCH ITEM ERROR]',repr(e),flush=True)
        matches.sort(key=lambda x:x['score'],reverse=True)
        return jsonify({'success':True,'matches':matches[:RESULT_LIMIT],'indexed_products':len(product_index),'search_mode':SEARCH_MODE,'query_design_features':q_features})
    except Exception as e:
        print('[SEARCH ERROR]',repr(e),flush=True)
        return jsonify({'error':'AI 의류 디자인 검색 중 오류가 발생했습니다.','detail':str(e)}),500


try:
    load_index(); load_anchor_cache()
except Exception as startup_error:
    print('[STARTUP ERROR]',repr(startup_error),flush=True)

if __name__=='__main__':
    port=int(os.environ.get('PORT','5000'))
    app.run(host='0.0.0.0',port=port,debug=False)
