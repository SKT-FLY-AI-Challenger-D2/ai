import os
import sys
import json
import asyncio
import aiohttp
from dotenv import load_dotenv
from google import genai
import requests
from typing import List, Dict
from bs4 import BeautifulSoup
import fitz
import io
import subprocess
import pytesseract
from PIL import Image
import re
import time
from google.genai.errors import APIError

from config import settings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import ModerationState, FactResult

# 환경 변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# ==========================================
# Speed knobs (추가)
# ==========================================
K_CLAIMS = 2                 # k=10 -> 2
SERPER_NUM = 2               # query당 결과 개수(기존 3)
MAX_URLS_PER_CLAIM = 3       # 명제당 스크래핑할 URL 상한
SCRAPE_CONCURRENCY = 8       # 본문 스크래핑 동시성 제한 (너무 크면 오히려 느려짐/차단)
PDF_OCR_MAX_PAGES = 1        # OCR 돌릴 최대 페이지 수 (0이면 OCR 비활성도 가능)


# 도메인별 공신력 화이트리스트
DOMAIN_WHITELISTS = {
    "HEALTH": "site:mohw.go.kr OR site:mfds.go.kr OR site:kdca.go.kr OR site:hira.or.kr",
    "FINANCE": "site:fss.or.kr OR site:fsc.go.kr OR site:bok.or.kr",
    "LEGAL_POLICY": "site:law.go.kr OR site:moj.go.kr OR site:mois.go.kr",
    "BEAUTY": "site:mfds.go.kr OR site:kcia.or.kr",
    "PRODUCT_TECH": "site:kcc.go.kr OR site:rra.go.kr OR site:kca.go.kr",
    "EDUCATION": "site:moe.go.kr OR site:moel.go.kr OR site:keis.or.kr",
    "GENERAL": "site:yna.co.kr OR site:kpf.or.kr"
}

# 공신력 도메인 확장 (정부, 공공기관, 교육기관, 국가기간 뉴스)
TRUST_DOMAINS = "site:go.kr OR site:or.kr OR site:ac.kr OR site:yna.co.kr"


# ==========================================
# 0. 데이터 전처리 (Preprocessing - graph.py Input → 내부 포맷)
# ==========================================
def data_preprocessing(state: ModerationState) -> ModerationState:
    """
    graph.py에서 전달받은 state의 input_text를 
    analyze_script()가 사용하기 적합한 형태로 파싱합니다.
    """
    print(f"\n{'='*20} 0. DATA PREPROCESSING STEP {'='*20}")
    
    raw_text = state.input_text
    
    if not raw_text or len(raw_text.strip()) < 20:
        print("[!] 입력 텍스트가 비어있거나 너무 짧습니다.")
        return state
    
    # 앞뒤 공백 제거 및 연속 공백/줄바꿈 정리
    cleaned = re.sub(r'\s+', ' ', raw_text.strip())
    
    state.input_text = cleaned
    
    return state

# ==========================================
# 1. 분석 노드 (Analysis Node)
# ==========================================
def analyze_script(state: ModerationState) -> ModerationState:
    """
    LLM을 사용하여 광고 스크립트의 도메인을 분류하고 명제 및 키워드를 추출합니다.
    """
    print(f"\n{'='*20} 1. AD ANALYSIS STEP {'='*20}")
    
    if not GOOGLE_API_KEY:
        print("[!] Error: GOOGLE_API_KEY is missing.")
        return state

    client = genai.Client(api_key=GOOGLE_API_KEY)

    # 수정된 시스템 프롬프트: 구체적 명제 추출 지시 및 키워드 최적화
    system_prompt = """
## Role
너는 유튜브 광고의 허위 사실과 사기성 콘텐츠를 적발하는 전문 '팩트 체크 분석 에이전트'다.

## Constraints
1. 분석 단계: 도메인 분류 -> 명제 추출 -> 검색 키워드 생성.
2. **도메인 분류 기준**: 다음 7가지 카테고리 중 가장 적합한 코드를 선택하라.
   - HEALTH : 의료 / 건강기능식품
   - FINANCE : 금융 / 투자
   - LEGAL_POLICY : 생활 / 법률 / 정부지원
   - BEAUTY : 뷰티
   - PRODUCT_TECH : IT / 전자기기
   - EDUCATION : 취업 / 교육 / 부업
   - GENERAL : 실시간 뉴스·기사 (사실 발생 여부 확인)
3. **명제 추출(Claim Extraction) 상세 지침**:
   - 수치, 인물, 기관, 효능 등 구체적인 팩트가 포함된 서술형으로 작성하라.
   - **명제는 최대 10개 이내로 제한한다. 즉, k <= 10.** 불필요하거나 중복된 명제는 제외하라.
4. **검색 키워드(Search Queries) 생성 전략 (Ultimate Version)**:
   - 각 명제당 3개의 키워드는 '입체적 검증'을 위해 서로 다른 출처를 타겟팅한다.
   - **Type 1 (공식/법적 증거)**: [공신력 있는 기관명] + [명제 키워드] + ["보도자료" OR "공식 입장" OR "가이드라인"]
     (예: 식품의약품안전처 레몬물 간해독 보도자료, 금융감독원 300% 수익률 주의보)
   - **Type 2 (비판적/사회적 검증)**: [명제 수치/약속] + ["사기" OR "허위" OR "적발" OR "실체"] + [2026]
     (예: "7일 만에 완치" 가짜뉴스 적발 2026, "원금 보장" 유튜브 광고 사기 실체)
   - **Type 3 (전문가/학술적 반박)**: [핵심 용어] + ["의학적 근거 부족" OR "논란" OR "팩트체크"]
     (예: 음주 전 우유 위벽보호 의학적 근거 부족, 주식 리딩방 수익률 조작 수법)
   - 명제에 인물(예: 김강현)이 포함된 경우, 3개 중 1개는 반드시 해당 인물의 [실존 여부]나 [사칭 사례]를 추적하는 쿼리로 구성하라.
 5. 

## Output Format (Strict JSON)
{
  "main_domain": "DOMAIN_CODE",
  "total_claims_count": k,
  "claims": [
    {
      "claim_id": 1,
      "claim_text": "상세한 명제 문장",
      "sub_domain": "DOMAIN_CODE",
      "priority": 1,
      "search_keywords": ["검증용 쿼리 1", "역추적용 쿼리 2", "권위 확인용 쿼리 3"]
    }
  ]
}
"""
    for model_name in settings.MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=f"{system_prompt}\n\n텍스트:\n{state.input_text}",
                config={"response_mime_type": "application/json"}
            )

            raw_json_text = response.text.strip()
            analysis_data = json.loads(raw_json_text)

            # ✅ (추가) 모델이 규칙을 안 지킬 때를 대비해 강제 컷
            claims = analysis_data.get("claims", []) or []
            analysis_data["claims"] = claims[:K_CLAIMS]
            analysis_data["total_claims_count"] = len(analysis_data["claims"])

            evidence_packet = [f"MAIN_DOMAIN: {analysis_data.get('main_domain')}"]
            for claim in analysis_data.get("claims", []):
                evidence_packet.append(json.dumps(claim, ensure_ascii=False))

            state.fact = FactResult(
                fake_score=0.0,
                fake_evidence=evidence_packet
            )

            return state

        except APIError as e:
            print(f"{model_name} API 에러(트래픽 등): {e}. 다음 모델 시도.")
            continue


    print(f"[!] Analysis Error: {e}")
    state.fact = FactResult(fake_score=0.0, fake_evidence=[f"Error: {str(e)}"])
    
    return state


# ==========================================
# 2. 정보 검색 노드 (Information Search Node)
# ==========================================
async def fetch_serper(session: aiohttp.ClientSession, query: str) -> List[Dict]:
    url = "https://google.serper.dev/search"
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    payload = json.dumps({"q": query, "gl": "kr", "hl": "ko", "num": SERPER_NUM})  # ✅ num 조절
    
    try:
        async with session.post(url, headers=headers, data=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result.get('organic', [])
            return []
    except Exception:
        return []


async def search_evidence_task(state: ModerationState) -> ModerationState:
    print(f"\n{'='*20} 2. AS_SYNC INFORMATION SEARCH STEP {'='*20}")
    if not SERPER_API_KEY or not state.fact:
        return state

    main_domain = state.fact.fake_evidence[0].replace("MAIN_DOMAIN: ", "")
    whitelist = DOMAIN_WHITELISTS.get(main_domain, "site:go.kr")
    claims_json = state.fact.fake_evidence[1:]
    
    search_tasks = []
    task_metadata = []

    # 결과 그룹화 데이터 초기화
    collected_data = {json.loads(c)['claim_id']: {"text": json.loads(c)['claim_text'], "results": []} for c in claims_json}

    connector = aiohttp.TCPConnector(ssl=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        for claim_str in claims_json:
            claim = json.loads(claim_str)
            
            # 1. 미정의 변수 queries를 refined_queries로 일치시킴
            refined_queries = [
                f"{claim['search_keywords'][0]} {whitelist}",
                claim['search_keywords'][1],
                f"{claim['search_keywords'][2]} filetype:pdf"
            ]

            for q in refined_queries:
                search_tasks.append(fetch_serper(session, q))
                task_metadata.append({"id": claim['claim_id'], "query": q})

        print(f"[*] 총 {len(search_tasks)}개의 비동기 쿼리 전송 중...")
        all_results = await asyncio.gather(*search_tasks)

        # 2. 결과 처리부
        for idx, results in enumerate(all_results):
            meta = task_metadata[idx]
            c_id = meta["id"]
            
            if results:
                for res in results:
                    # 필터링 없이 수집된 결과 그대로 저장
                    collected_data[c_id]["results"].append({
                        "title": res.get("title"),
                        "link": res.get("link"),
                        "snippet": res.get("snippet")
                    })

    final_packet = [f"MAIN_DOMAIN: {main_domain}"]
    for c_id, data in collected_data.items():
        final_packet.append(json.dumps({
            "claim_id": c_id,
            "claim_text": data["text"],
            "collected_info": data["results"]
        }, ensure_ascii=False))

    state.fact.fake_evidence = final_packet
    print(f"[+] 정보 수집 및 콘솔 출력 완료.")
    return state

def search_evidence(state: ModerationState) -> ModerationState:
    # 비동기 함수를 동기 루프에서 실행하기 위한 장치
    return asyncio.run(search_evidence_task(state))


# ==========================================
# 3. 본문 스크래핑 노드 (Content Scraping Node)
# ==========================================
async def fetch_full_text(session: aiohttp.ClientSession, url: str) -> str:
    """HTML과 PDF를 모두 처리하여 순수 본문 텍스트를 추출합니다."""
    fitz.TOOLS.mupdf_display_errors(False)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    }

    try:
        async with session.get(url, headers=headers, timeout=20, allow_redirects=True, ssl=False) as response:
            if response.status != 200:
                return f"Fail: Status Code {response.status}"

            content_bytes = await response.read()
            if not content_bytes:
                return "Fail: Empty content"

            content_type = response.headers.get('Content-Type', '').lower()

            # [CASE 1] PDF 처리
            if 'application/pdf' in content_type or content_bytes.startswith(b'%PDF'):
                try:
                    doc = fitz.open(stream=io.BytesIO(content_bytes), filetype="pdf")
                    full_text = ""

                    # ✅ (추가) OCR 포함 최대 페이지 제한
                    max_pages = min(len(doc), max(1, PDF_OCR_MAX_PAGES))
                    for i in range(max_pages):
                        page = doc.load_page(i)
                        page_text = (page.get_text() or "").strip()

                        is_korean = any('\uac00' <= c <= '\ud7a3' for c in page_text)
                        if is_korean and len(page_text) > 80:
                            full_text += page_text + " "
                        else:
                            # OCR 비활성화 옵션: PDF_OCR_MAX_PAGES <= 0 같은 방식도 가능
                            if PDF_OCR_MAX_PAGES <= 0:
                                continue
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                            img = Image.open(io.BytesIO(pix.tobytes()))
                            ocr_text = pytesseract.image_to_string(img, lang='kor+eng')
                            full_text += ocr_text + " "

                    doc.close()
                    full_text = " ".join(full_text.split())
                    if not full_text:
                        return "Fail: PDF empty or OCR disabled"
                    return "[PDF 전문 수집(OCR포함)] " + full_text
                except Exception as e:
                    return f"Fail: PDF/OCR Error ({str(e)})"

            # [CASE 2] HWP
            elif 'application/x-hwp' in content_type or 'hwp' in url.lower() or content_bytes.startswith(b'\xd0\xcf\x11\xe0'):
                try:
                    temp_hwp_path = f"temp_{os.getpid()}_{hash(url)}.hwp"
                    with open(temp_hwp_path, "wb") as f:
                        f.write(content_bytes)
                    process = subprocess.run(
                        ['hwp5txt', temp_hwp_path],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='ignore'
                    )
                    extracted_text = process.stdout
                    if os.path.exists(temp_hwp_path):
                        os.remove(temp_hwp_path)
                    return "[HWP 전문 수집됨] " + " ".join((extracted_text or "").split())
                except Exception as e:
                    return f"Fail: HWP Parsing Error ({str(e)})"

            # [CASE 3] HTML
            else:
                try:
                    try:
                        html = content_bytes.decode('cp949')
                    except UnicodeDecodeError:
                        html = content_bytes.decode('utf-8', errors='ignore')

                    soup = BeautifulSoup(html, 'html.parser')
                    for el in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                        el.decompose()

                    clean_text = " ".join(soup.get_text(separator=' ').split())
                    return clean_text if clean_text else "Fail: No readable text found"
                except Exception as e:
                    return f"Fail: HTML Processing Error ({str(e)})"
    except Exception as e:
        return f"Fail: {str(e)}"


async def scrape_evidence_task(state: ModerationState) -> ModerationState:
    print(f"\n{'='*20} 3. FULL CONTENT SCRAPING STEP {'='*20}")

    if not state.fact or not state.fact.fake_evidence:
        return state

    main_domain = state.fact.fake_evidence[0].replace("MAIN_DOMAIN: ", "")
    claims_json = state.fact.fake_evidence[1:]

    # ✅ (추가) 명제별 URL을 "공신력 우선"으로 제한 수집
    all_urls = []
    per_claim_urls = {}  # claim_id -> [urls]

    for claim_str in claims_json:
        claim_data = json.loads(claim_str)
        c_id = claim_data.get("claim_id")
        urls = []

        for res in claim_data.get("collected_info", []):
            link = res.get("link")
            if link and link.startswith("http"):
                urls.append(link)

        # 중복 제거(명제 내부)
        urls = list(dict.fromkeys(urls))

        # 공신력 점수로 정렬(낮을수록 공신력 높음)
        urls.sort(key=get_source_authority_score)

        # ✅ 상위 N개만 스크래핑
        urls = urls[:MAX_URLS_PER_CLAIM]
        per_claim_urls[c_id] = urls
        all_urls.extend(urls)

    # ✅ (추가) 전체 중복 제거
    all_urls = list(dict.fromkeys(all_urls))

    print(f"[*] 스크래핑 대상 URL: 총 {len(all_urls)}개 (명제당 최대 {MAX_URLS_PER_CLAIM})")

    connector = aiohttp.TCPConnector(ssl=False)

    # ✅ (추가) 동시성 제한 세마포어
    sem = asyncio.Semaphore(SCRAPE_CONCURRENCY)

    async def bounded_fetch(url: str) -> str:
        async with sem:
            return await fetch_full_text(session, url)

    async with aiohttp.ClientSession(connector=connector) as session:
        scrape_tasks = [bounded_fetch(url) for url in all_urls]
        contents = await asyncio.gather(*scrape_tasks)

    url_to_content = dict(zip(all_urls, contents))

    final_packet = [f"MAIN_DOMAIN: {main_domain}"]
    for claim_str in claims_json:
        claim_data = json.loads(claim_str)
        c_id = claim_data.get("claim_id")

        # ✅ (추가) 제한된 URL만 full_text 붙이기
        allowed = set(per_claim_urls.get(c_id, []))

        for res in claim_data.get("collected_info", []):
            res_link = res.get("link")
            if res_link in allowed:
                res["full_text"] = url_to_content.get(res_link, "No content retrieved.")
            else:
                # 스크래핑 안 한 것들은 snippet만 활용하도록 표시
                res["full_text"] = "Skipped: not in top-N URLs for this claim."

        final_packet.append(json.dumps(claim_data, ensure_ascii=False))

    state.fact.fake_evidence = final_packet
    print(f"[+] 본문 수집 완료. (총 {len(all_urls)}건)")
    return state

def scrape_evidence(state: ModerationState) -> ModerationState:
    return asyncio.run(scrape_evidence_task(state))


# ==========================================
# 4. 검증 노드 (Verification Node)
# ==========================================

def get_source_authority_score(url: str) -> int:
    """
    URL을 기반으로 공신력 점수를 반환합니다. 점수가 낮을수록 공신력이 높습니다.
    1점: 정부/공공기관 (.go.kr, .or.kr 등)
    2점: 논문/학술 (.ac.kr, scholar.google 등)
    3점: 뉴스/언론 (yna.co.kr, 주요 언론사)
    4점: 일반 웹사이트
    5점: SNS/블로그 (youtube, facebook, post.naver 등)
    """
    domain = url.lower()
    
    # Priority 1: Government/Public
    if any(x in domain for x in [".go.kr", ".or.kr", "who.int", "un.org", "korea.kr"]):
        return 1
        
    # Priority 2: Academic
    if any(x in domain for x in [".ac.kr", "scholar.google", "riss.kr", "dbpia.co.kr", "ier.snu.ac.kr"]):
        return 2
        
    # Priority 3: News (Major domains)
    news_domains = [
        "yna.co.kr", "chosun.com", "joongang.co.kr", "donga.com", "hani.co.kr", 
        "khan.co.kr", "kmib.co.kr", "mk.co.kr", "hankyung.com", "imbc.com", 
        "kbs.co.kr", "sbs.co.kr", "ytn.co.kr", "newsis.com", "news1.kr"
    ]
    if any(x in domain for x in news_domains) or "news" in domain:
        return 3
        
    # Priority 5: SNS/Blogs (Explicit low priority)
    sns_domains = [
        "youtube.com", "youtu.be", "facebook.com", "instagram.com", "twitter.com", 
        "tiktok.com", "blog.naver.com", "tistory.com", "brunch.co.kr", "dcinside.com", 
        "fmkorea.com", "ppomppu.co.kr", "theqoo.net"
    ]
    if any(x in domain for x in sns_domains):
        return 5
        
    # Priority 4: General Web (Default)
    return 4

def verify_facts(state: ModerationState) -> ModerationState:
    """
    수집된 full_text 증거를 각 명제에 매핑하여 진위 여부를 최종 판정합니다.
    """
    print(f"\n{'='*20} 4. FACT VERIFICATION STEP {'='*20}")
    
    if not state.fact or len(state.fact.fake_evidence) <= 1:
        print("[!] 검증할 증거 데이터가 부족합니다.")
        return state

    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    # 1. 데이터 분리 및 초기화
    main_domain = state.fact.fake_evidence[0]
    claims_json = state.fact.fake_evidence[1:]
    
    verified_claims = []
    total_risk_score = 0.0

    # 2. 명제별 순회 및 판정
    for claim_str in claims_json:
        claim = json.loads(claim_str)
        c_id = claim.get("claim_id")
        claim_text = claim.get("claim_text")
        collected_info = claim.get("collected_info", [])

        # 명제와 full_text 매핑: 한 명제에 연결된 여러 소스를 하나의 컨텍스트로 결합
        evidence_context = ""
        for info in collected_info:
            source_title = info.get("title", "제목 없음")
            source_url = info.get("link", "URL 없음")
            # 스크래핑 실패 시 snippet이라도 활용, 성공 시 full_text 사용
            body = info.get("full_text", "")
            if body.startswith("Fail:") or len(body) < 50:
                body = f"[Snippet 정보]: {info.get('snippet', '정보 없음')}"
            
            evidence_context += f"\n---\n[출처: {source_title}]\n[URL: {source_url}]\n[본문]: {body[:3000]}\n"

        # LLM 판정 프롬프트 (0.0~1.0 연속 점수제 적용 + 0.0/1.0 제외 및 1문장 제한)
        verification_prompt = f"""
## Role
너는 수집된 웹 본문 자료를 바탕으로 광고의 진위 여부를 분석하는 전문 팩트체커다.

## Task
제공된 [근거 자료]를 바탕으로 [대상 명제]의 위험도(Risk Score)를 판정하라.
**절대로 0.0점(완벽한 진실)이나 1.0점(완벽한 허위)은 부여하지 말라.** 현실 세계에서 100% 확신은 불가능하기 때문이다.

[대상 명제]
{claim_text}

[근거 자료]
{evidence_context}

## 판정 가이드라인 (Risk Score: 0.0 ~ 1.0)
- **0.0 ~ 0.2 (매우 낮음)**: 신뢰할 수 있는 공식 근거가 명확하며, 사실로 판단됨.
- **0.3 ~ 0.5 (주의)**: 일부 근거는 있으나 과장된 표현이 섞여 있거나, 근거가 불충분함.
- **0.6 ~ 0.8 (의심)**: 신뢰할 수 있는 반박 자료가 존재하거나, 전형적인 과대광고 패턴을 보임.
- **0.9 ~ 1.0 (매우 높음)**: 공식 기관에 의해 허위/사기로 적발되었거나, 명백한 거짓 정보임.
- 단, 0.0과 1.0은 절대로 부여하지 말 것

## Output Requirements (Very Important!)
1. **reason (분석 내용)**: 
   - 판정 이유를 객관적인 뉘앙스로 설명하되, **반드시 '~니다.'로 끝나는 정중한 문장**으로 작성하라.
   - 예시: "해당 명제는 의학적 근거가 부족하며, 관련 기관에서도 주의를 당부한 바 있습니다."
2. **source_name (출처명)**:
   - 근거 자료에서 확인된 경우, 가능한 한 **"기관명 - 제목 | 사이트명"** 형식으로 추출하라.
   - 예시: "식품의약품안전처 - 레몬물 효능 팩트체크 | 올바른 건강 정보", "질병관리청 - 음주 가이드라인 | 국가건강정보포털"

## Output Format (Strict JSON)
{{
  "risk_score": 0.01~0.99 사이의 실수,
  "reason": "분석 내용 (반드시 '~니다.'로 종료)",
  "source_name": "출처명 (기관명 - 제목 | 사이트명)",
  "concise_summary": "핵심 반박/검증 내용을 한 문장으로 요약 (ex. 바세린의 수면 개선 효과를 뒷받침하는 어떠한 의학적, 과학적 근거도 찾을 수 없습니다.)"
  "evidence_quote": "본문에서 근거가 된 핵심 문구 직접 인용 (없으면 빈 문자열)",
  "evidence_url": "해당 근거가 포함된 원본 URL"
}}
"""

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=verification_prompt,
                config={"response_mime_type": "application/json"}
            )
            
            result = json.loads(response.text)
            
            # 1. raw_score 추출 및 보정 (0.01~0.99 범위 강제 클리핑)
            raw_score = float(result.get("risk_score", 0.5))
            risk_score = max(0.01, min(0.99, raw_score))
            
            # 결과 병합
            claim.update({
                "risk_score": risk_score,
                "reason": result.get("reason"),
                "source_name": result.get("source_name", "확인된 출처"),
                "concise_summary": result.get("concise_summary", "요약 정보 없음"),
                "evidence_quote": result.get("evidence_quote"),
                "evidence_url": result.get("evidence_url")
            })

            total_risk_score += risk_score
            verified_claims.append(claim)

        except Exception as e:
            print(f"  [!] 명제 {c_id} 판정 중 오류 발생: {e}")
            claim.update({"risk_score": 0.5, "reason": f"분석 오류로 인한 중간값 배정: {str(e)}"})
            total_risk_score += 0.5
            verified_claims.append(claim)

    # 3. 최종 스코어 및 State 업데이트
    num_claims = len(verified_claims)
    # 명제별 위험도의 평균값 계산
    final_fake_score = (total_risk_score / num_claims) if num_claims > 0 else 0.0

    final_packet = [main_domain]
    for c in verified_claims:
        final_packet.append(json.dumps(c, ensure_ascii=False))
        
    state.fact.fake_score = final_fake_score
    state.fact.fake_evidence = final_packet
    
    print(f"\n[+] 전 명제 검증 완료. 최종 평균 위험도 점수: {final_fake_score:.4f}")
    return state

# ==========================================
# 5. 리포트 노드 (데이터 후처리 - 내부 포맷 → graph.py Output)
# ==========================================
def generate_final_report(state: ModerationState) -> ModerationState:
    """
    내부 파이프라인의 결과를 graph.py/reporter_node가 기대하는 
    출력 형식으로 변환합니다.
    
    변환 내용:
    - fake_score: 이미 0.0~1.0 이므로 그대로 유지 (반올림만 수행)
    - fake_evidence: JSON 문자열 리스트 → 점수와 사유가 포함된 가독성 있는 리스트
    """
    print(f"\n{'='*20} 5. FINAL REPORT (POST-PROCESSING) {'='*20}")
    
    if not state.fact:
        print("[!] fact 결과가 없습니다. 후처리를 스킵합니다.")
        return state
    
    # 1. Score 최종 정리 (이미 0.0~1.0이므로 소수점 4자리 반올림)
    final_score = round(state.fact.fake_score, 4)
    print(f"  [최종 위험도 점수]: {final_score:.4f}/1.0")
    
    # 2. Evidence 변환
    raw_evidence = state.fact.fake_evidence
    claims_data = []
    main_domain_str = "MAIN_DOMAIN: GENERAL"
    
    for item in raw_evidence:
        if item.startswith("MAIN_DOMAIN:"):
            main_domain_str = item
            continue
        try:
            claims_data.append(json.loads(item))
        except (json.JSONDecodeError, AttributeError):
            continue

    # 정렬 로직 변경: 1순위 - 공신력 점수(낮을수록 좋음), 2순위 - 위험도(높을수록 중요)
    sorted_claims = sorted(
        claims_data, 
        key=lambda x: (
            get_source_authority_score(x.get('evidence_url', '')), 
            -x.get('risk_score', 0.0)
        )
    )
    
    top_two_claims = sorted_claims[:2]

    readable_evidence = []
    
    for claim in top_two_claims:
        concise_summary = claim.get("concise_summary", "요약 정보 없음")
        readable_evidence.append(concise_summary.strip())

    # 3. State 업데이트
    state.fact.fake_score = final_score
    state.fact.fake_evidence = readable_evidence
    
    print(f"\n[+] 후처리 완료. graph.py로 반환 준비 완료.")
    return state

# ==========================================
# LangGraph 노드 인터페이스 (graph.py 연동)
# ==========================================
def fact_check_node(state: ModerationState) -> dict:
    """
    LangGraph 노드 래퍼 함수.
    graph.py에서 호출되며, legal_node/detector_node와 동일한 dict 반환 패턴을 따릅니다.
    내부적으로 5단계 팩트체크 파이프라인을 순차 실행합니다.
    """
    
    start_time_str = time.strftime("%H:%M:%S")
    print(f"--- [Fact Check Node] 시작 시각: {start_time_str} ---")
    
    state = data_preprocessing(state)   # 0. 데이터 전처리  
    state = analyze_script(state)       # 1. 스크립트 분석
    state = search_evidence(state)      # 2. 정보 검색
    state = scrape_evidence(state)      # 3. 정보 수집
    state = verify_facts(state)         # 4. 검증
    state = generate_final_report(state) # 5. 리포트 생성
    
    end_time_str = time.strftime("%H:%M:%S")
    print(f"--- [Fact Check Node] 종료 시각: {end_time_str} ---")
    
    return {"fact": state.fact}


# ==========================================
# 테스트용 메인 오케스트레이터 (Orchestrator)
# ==========================================
def run_fact_check_pipeline(raw_script: str):
    # 0. Input 정제

    state = ModerationState(input_text=raw_script)
    
    state = analyze_script(state)  # 1. 스크립트 분석    
    state = search_evidence(state)  # 2. 정보 검색
    state = scrape_evidence(state)  # 3. 정보 수집
    state = verify_facts(state)     # 4. 검증
    state = generate_final_report(state)   # 5. 리포트 생성
    
    # graph.py에게 전달할 최종 보고서 미리보기
    print(f"\n{'='*20} PIPELINE EXECUTION FINISHED {'='*20}")
    print(f"\n{'='*20} [graph.py 전달용 최종 Output] {'='*20}")
    if state.fact:
        print(f"  fake_score: {state.fact.fake_score}")
        print(f"  fake_evidence ({len(state.fact.fake_evidence)}건):")
        for i, ev in enumerate(state.fact.fake_evidence):
            print(f"    [{i+1}] {ev}")
    else:
        print("  [!] fact 결과 없음")
    print(f"{'='*60}")
    
    return state

if __name__ == "__main__":
    sample_ad = """
여러분, 술을 꼭 끊지 않아도 건강을 지킬 수 있다면 믿으시겠습니까? 안녕하세요. 소화기내과 김강현입니다. 나이가 들수록 술은 단순한 기호품이 아니라 삶의 한 부분이 됩니다. 친구와의 만남, 하루를 마무리하는 작은 의식, 때로는 외로움을 달래주는 벗처럼 자리 잡았죠. 그래서 술을 무조건 끊어야 한다는 말이 솔직히 와닿지 않는 분들이 많습니다. 하지만 문제는 술 자체가 아니라 세월 속에서 달라진 우리의 몸입니다. 예전에는 아무렇지 않게 마셨던 양도 이제는 몇 잔만 마셔도 다음 날 오후까지 술이 깨지지 않아 고생하는 경우가 많습니다. 그런데 희망은 있습니다. 실제로 70대 어르신 한 분은 오늘 이 영상에서 알려드리는 원칙 몇 가지만 지켰을 뿐인데 불과 3주 만에 간 수치가 정상으로 돌아왔습니다. 지금은 예전처럼 술을 즐기셔도 다음 날 힘들지 않고 아침에 상쾌하게 일어나 산책을 하고 손주와 놀 만큼 활력을 되찾으셨습니다. 여러분도 충분히 그렇게 하실 수 있습니다. 오늘 제가 알려드릴 몇 가지 단순한 원칙만 지키신다면은 술을 즐기면서도 간, 심장, 뇌까지 건강을 지킬 수 있습니다. 반대로 이 원칙을 무시한다면 몸속에서 어떤 무서운 변화가 일어나는지 술 없이는 잠들지 못해 뇌 건강까지 해치는 위험한 악순환이 어떻게 생기는지도 싹 다 알려드리겠습니다. 시작에 앞서 구독과 좋아요 알림 설정까지 꼭 눌러주시고 끝까지 시청해주세요. 이 영상이 단순히 술 이야기가 아니라 여러분의 건강한 노후를 지켜줄 전환점이 될 수 있기 때문입니다. 자, 첫 번째 원칙부터 말씀드리겠습니다. 술은 어떻게 마시느냐도 중요하지만 사실 술잔을 들기 전에 몸을 어떻게 준비했느냐가 훨씬 더 큰 차이를 만듭니다. 준비 없이 마시는 술은 우리 몸을 불시에 덮치는 불청객 같아서 간이 당황하고 위가 상처를 입게 돼요. 특히 속이 비어 있을 때 술을 마시는 습관은 정말 위험합니다. 서울대병원 연구에 따르면 술을 완전히 끊지 않아도 단순히 마시는 습관만 바꿔도 우리 몸에 해로운 영향을 크게 줄일 수 있다는 결과가 있습니다. 그러니 희망은 충분히 있는 겁니다. 많은 분들이 이런 경험 있으셨을 겁니다. 친구분들과 만나자마자 밥도 안 먹고 맥주나 소주부터 들이킨 날 말이에요. 그때는 술이 빨리 돌고 기분이 금세 좋아지는 것 같지만 잠시뿐이죠. 얼굴이 화끈 달아오르고 속은 불편하고 다음 날 아침까지 머리가 지끈거려서 하루 종일 힘들었던 적 없으셨나요? 어 사실 이게 다 이유가 있습니다. 속이 비어 있을 때 술을 마시면 술이 위에 머물지 않고 곧장 몸속으로 흡수됩니다. 그래서 알코올이 평소보다 두세 배는 빠르게 퍼지게 되죠. 간은 갑작스럽게 몰려드는 알코올을 처리하느라 과부하에 걸리고 위벽은 상처를 입어 속 쓰림과 구토, 어지럼증까지 이어질 수 있습니다. 겉으론 멀쩡해 보여도 몸속에서는 이미 작은 폭풍이 시작되는 셈입니다. 그렇다면 어떻게 해야 할까요? 사실 방법은 어렵지 않습니다. 술을 마시기 전에 간단하게라도 속을 채워주는 겁니다. 삶은 달걀 하나, 바나나 하나, 치즈 한 장, 우유 한 컵이나 요거트 중 하나면 충분합니다. 이런 간단한 음식들이 위벽을 보호하고 술이 흡수되는 속도를 늦춰줍니다. 특히 치즈나 우유처럼 단백질과 지방이 들어있는 음식은 술이 주는 자극을 막아주는 든든한 방패가 됩니다. 실제로 제 환자 중 한 분은 술만 마시면 속이 쓰리고 다음 날 힘들어 늘 약에 의지하셨는데요. 제가 술 드시기 전에 치즈 한 장만이라도 꼭 드셔 보세요라고 권해드렸습니다. 놀랍게도 그 뒤로는 다음 날 아침이 훨씬 가볍고 속도 덜 불편해졌다고 하셨어요. 작은 습관 하나가 몸 전체를 지켜주는 힘이 된 겁니다. 여기서 또 중요한 것이 있습니다. 바로 물입니다. 술이 몸에 들어오면 간은 알코올을 해독하기 위해 엄청난 양의 수분을 사용합니다. 물을 챙기지 않고 술만 마시면 몸은 빠르게 탈수에 빠지고 두통, 갈증, 피로가 몰려오게 됩니다. 그래서 술을 마시기 전 미리 물을 마셔 두는 게 꼭 필요합니다. 특히 찬물보다는 미지근한 물이 더 좋아요. 위에 부담을 덜 주고 흡수도 훨씬 빠르기 때문입니다. 여러분, 술은 단순히 마시는 순간만 중요한 게 아닙니다. 술잔을 들기 전에 이 작은 준비가 그날 밤의 몸 상태 그리고 다음 날의 하루를 완전히 바꿀 수 있습니다. 오늘 저녁부터라도 이 습관을 시작해보세요. 내일 아침에 몸이 얼마나 달라지는지 직접 느끼실 수 있을 겁니다. 오늘 내용 전반에서 비슷한 이야기가 반복되어 들리실 수도 있습니다. 하지만 그것은 그만큼 중요한 부분이기 때문에 다시 한번 강조 드리는 것이니 꼭 기억해 두시기 바랍니다. 자, 이제 준비가 끝났습니다. 하지만 술은 준비만 잘한다고 안전한 게 아닙니다. 술을 마시는 그 순간에도 지켜야 할 원칙이 따로 있습니다. 지금부터는 술을 마시는 중에 반드시 실천해야 할 방법들을 하나씩 알려드리겠습니다. 여러분, 술을 마실 때 가장 큰 문제는 바로 속도입니다. 술은 빨리 마실수록 간이 따라가지 못하고 남은 알코올이 그대로 혈액을 타고 퍼져 온몸을 괴롭힙니다. 얼굴이 벌겋게 달아오르고 속이 울렁거리고 머리가 지끈거리는 경험, 아마 누구나 해보셨을 겁니다. 그런데도 우리는 자꾸 분위기에 휩쓸려 원샷을 하거나 잔을 연달아 비우곤 하지요. 그래서 제가 드리고 싶은 말씀은 단순합니다. 술은 천천히 마셔야 합니다. 한 잔에 최소 30분 정도 여유를 두면 간이 숨을 고르고 알코올을 처리할 시간을 벌 수 있습니다. 하지만 혹시 이렇게 생각하실 수도 있습니다. 술자리가 빠르게 돌아가는데 혼자만 느리게 마실 수 있나 하는 걱정 말입니다. 맞습니다. 현실적으로 그럴 수 있지요. 그래서 대안이 있습니다. 술을 조금 빨리 마셔야 하는 상황이라면 반드시 물을 곁들이는 겁니다. 술 한 잔을 마셨다면 그다음엔 꼭 물 한 잔을 더하세요. 그러면은 술이 우리 몸속에서 희석되고 알코올이 퍼지는 속도도 훨씬 느려집니다. 마치 뜨거운 국물에 물을 조금 부어 식히듯 물이 술의 독성을 완화해주는 것이지요. 이 습관 하나만 실천해도 다음 날 숙취가 줄어들고 간이 훨씬 수월해집니다. 또 중요한 건 안주입니다. 술만 들이키면 우리 몸은 금방 힘들어집니다. 하지만 치즈나 땅콩, 삼겹살처럼 적당한 지방이 들어있는 음식을 함께 먹으면 위벽이 보호되어 알코올이 흡수되는 속도가 느려집니다. 술자리를 즐기는 것 같지만 사실은 몸을 지키는 지혜로운 선택이 되는 겁니다. 그리고 꼭 기억해야 할 것이 있습니다. 술자리 전부터 오늘 마실 양을 정해두는 겁니다. 소주 반 병까지만, 맥주 두 잔까지만. 이런 기준이 있어야 분위기에 휩쓸리지 않습니다. 술잔을 늘 조금씩 남겨두는 것도 좋은 방법입니다. 잔이 비어 있으면 주변에서 채워 넣지만 잔에 술이 조금이라도 남아 있으면 그만큼 권유도 줄어들기 때문입니다. 술은 천천히 마시는 것만으로도 몸의 부담이 크게 줄어듭니다. 여기에 물을 곁들이고 안주를 현명하게 선택하는 습관까지 더해지면 술자리는 전혀 다른 결과를 가져옵니다. 그리고 이렇게만 해도 몸은 훨씬 편안해지지만 사실 진짜 변화는 그 다음 단계에서 더 크게 나타납니다. 술잔을 내려놓은 후 어떻게 회복하느냐에 따라 내일 하루가 완전히 달라지기 때문입니다. 여러분, 여기서 안타까운 사례 하나를 말씀드리고 넘어가야 할 것 같습니다. 경기도 부천에 사시던 유 사장님은 식당 재료 도매상 일을 하셨습니다. 낮에는 늘 바쁘게 뛰어다니셨지만 저녁만 되면 혼자 나와 주문을 받으며 술을 즐기셨습니다. 특히 막걸리를 몸에 좋은 술이라고 믿으셔서 하루도 빠짐없이 드셨고 늘 하시던 말씀이 세상에서 술이 제일 맛있다는 거였습니다. 하지만 그렇게 몇 년간 매일 술을 마신 결과 간 수치가 걷잡을 수 없이 악화되었고 결국 건강을 회복하지 못한 채 세상을 떠나셨습니다. 가족분들 말씀에 따르면 술을 즐길 때는 늘 행복해 보였지만 마지막은 참 힘들고 안타까운 길이었다고 합니다. 이 사례가 보여주는 건 아주 분명합니다. 술 자체가 아니라 매일 마시는 습관이 몸을 망가뜨린다는 사실입니다. 그래서 오늘 말씀드리는 원칙 중 하나는 술을 즐기더라도 간에게 반드시 쉴 시간을 주어야 한다는 겁니다. 자, 이제 구체적으로 술을 마신 다음 날 어떻게 관리하면 우리 몸이 다시 회복할 수 있는지 그 루틴을 알려드리겠습니다. 여러분, 술자리는 끝났는데 몸은 여전히 술과 싸우고 있다는 사실 알고 계셨나요? 많은 분들이 술을 마시고 나면 그냥 잠들면 끝이라고 생각하시지만 사실 그때부터가 진짜 시작입니다. 우리 간은 밤새도록 알코올을 분해하느라 쉴 틈이 없고 몸속 수분과 영양소는 빠르게 고갈됩니다. 그래서 아침에 눈을 뜨면 머리가 무겁고 입은 바짝 마르고 속은 더부룩한 것이지요. 이때 중요한 것은 술 마신 다음 날을 어떻게 보내느냐입니다. 그 관리 하나로 몸의 회복 속도가 완전히 달라집니다. 첫 번째는 잠들기 전 물 한 컵입니다. 단순해 보이지만 아주 강력한 방법입니다. 우리 몸은 자는 동안에도 간이 계속 알코올을 해독하기 때문에 많은 수분이 필요합니다. 물을 마시지 않고 그냥 자면 밤새 탈수가 진행되어 아침에 머리가 지끈거리고 온몸이 무겁습니다. 하지만 잠들기 전 물 한 컵을 챙기면 다음 날 아침이 훨씬 가벼워집니다. 두 번째는 바로 눕지 않는 것입니다. 술 마신 후 피곤해서 바로 눕고 싶어도 최소한 30분은 몸을 정리해주는 것이 좋습니다. 가볍게 샤워를 하면 혈액 순환이 좋아지고 알코올이 빨리 분해되는 데 도움을 줍니다. 양치질도 꼭 필요합니다. 술 마신 뒤 입안 세균은 평소보다 몇 배나 늘어나는데 이를 방치하면 입 냄새와 잇몸 질환으로 이어질 수 있기 때문입니다. 세 번째는 아침 루틴입니다. 아침에 눈을 뜨자마자 가장 먼저 해야 할 일은 물 마시기입니다. 따뜻한 물에 레몬 한 조각을 넣어 마시면 간 해독에도 도움이 되고 잃어버린 전해질을 채워줍니다. 레몬이 없다면 그냥 미지근한 물이라도 충분합니다. 그리고 바나나 한 개를 곁들여 보세요. 바나나는 칼륨이 풍부해 술로 인해 생기는 붓기와 피로를 줄여줍니다. 여기에 종합 비타민까지 챙기면 더 확실한 도움이 됩니다. 네 번째는 무리하지 않는 하루를 보내는 것입니다. 술 마신 다음 날에는 격한 운동이나 과로는 오히려 회복을 늦춥니다. 대신 가볍게 산책을 하거나 스트레칭으로 몸을 풀어주면 훨씬 수월하게 회복할 수 있습니다. 술이 몸을 괴롭힌 시간만큼 회복에도 시간을 줘야 하는 것이지요. 여러분, 이렇게 술 마신 후 단순한 습관 몇 가지만 바꿔도 다음 날 몸 상태는 완전히 달라집니다. 그런데 여기서 끝이 아닙니다. 지금까지는 술과 몸의 관계였다면 다음에는 술과 약의 관계를 말씀드려야 합니다. 생각보다 많은 분들이 간과하는 부분인데 약과 술이 함께할 때는 훨씬 더 위험한 결과가 생길 수 있습니다. 그 이야기는 이어서 알려드리겠습니다. 여러분, 술을 아무리 지혜롭게 마신다고 해도 절대로 함께 해서는 안 되는 것들이 있습니다. 이 부분은 타협이 없는 절대 금기라고 생각하셔야 합니다. 먼저 술과 수면제는 절대 함께 하면 안 됩니다. 술은 뇌를 눌러서 졸음을 부르고 수면제도 같은 작용을 하지요. 그런데 이 둘이 합쳐지면 뇌가 너무 눌려서 숨 쉬는 힘마저 약해질 수 있습니다. 실제로 수면제와 술을 동시에 복용하다가 밤새 호흡이 불안정해져 응급실로 실려오는 경우가 종종 있습니다. 단순히 피곤하다고 술과 약을 같이 하는 건 정말 위험한 선택입니다. 다음은 술과 담배입니다. 술이 몸속을 열어두면 담배 속 독성 물질이 훨씬 더 빠르게 흡수돼요. 그래서 술자리에서 담배를 같이 하는 습관은 구강암이나 식도암 같은 암 위험을 몇 배나 높여버립니다. 흔히들 술에 담배가 잘 어울린다고 말하지만 사실은 서로의 해로움을 키우는 최악의 조합입니다. 또 술과 무리한 운동도 절대 금물입니다. 술이 아직 몸에 남아있는 상태에서 달리기를 하거나 무거운 물건을 들면 작은 충격에도 쉽게 균형을 잃고 넘어질 수 있어요. 특히 나이가 들수록 이런 낙상은 고관절 골절로 이어지기 쉬운데 한 번 골절되면 몇 달 동안 거동이 힘들어지고 삶의 질이 크게 떨어집니다. 마지막으로 술은 건강 문제를 넘어서 가족과 삶에도 깊은 파급 효과를 줍니다. 반복되는 음주는 기억력을 떨어뜨려 치매 위험을 높이고 성격 변화를 불러 가족과 갈등을 만들기도 하지요. 실제로 술 때문에 대화가 줄고 관계가 멀어져 고립감을 겪는 분들도 적지 않습니다. 술은 결국 나 혼자만의 문제가 아니라 곁에 있는 사람들까지 함께 영향을 받는다는 점을 꼭 기억해야 합니다. 그래서 제가 강조 드리는 건 아주 단순합니다. 술은 지혜롭게 마실 수 있지만 수면제와 감기약, 흡연, 무리한 운동 그리고 가족과의 갈등으로 이어질 정도의 과음만큼은 단 하나도 허용되지 않는다는 사실입니다. 그리고 여기서 궁금해지실 겁니다. 그렇다면은 술을 마셔도 괜찮은 경우는 어디까지일까요? 또 지켜야 할 최소한의 안전선은 무엇일까요? 그 답은 바로 다음장에서 알려드리겠습니다. 여러분, 술자리는 끝났지만 술의 영향은 쉽게 사라지지 않습니다. 아침에 눈을 떴을 때 머리가 무겁고 속이 울렁거리고 갈증 때문에 물을 찾게 되는 경험 다들 있으시죠? 이것은 알코올이 여전히 몸속에서 작용하고 있다는 신호입니다. 그렇다면 술을 마신 다음 날 어떻게 해야 몸을 빠르게 회복할 수 있을까요? 제가 환자분들께 꼭 알려드리는 다섯 가지 루틴을 말씀드리겠습니다. 첫째는 아침에 눈을 뜨자마자 물을 마시는 겁니다. 술은 몸속 수분을 빠르게 빼앗아 가기 때문에 전 날 부족해진 수분을 가장 먼저 채워주어야 합니다. 특히 따뜻한 물이나 레몬을 띄운 물은 간 해독을 돕고 위장을 편안하게 해줍니다. 물 한 컵이 단순해 보이지만 두통과 갈증을 크게 줄여주는 중요한 습관입니다. 둘째는 아침 식사에 바나나와 계란을 챙기는 겁니다. 술은 칼륨과 비타민B를 많이 소모시키는데 바나나는 칼륨이 풍부해서 붓기를 줄여주고 근육 경련을 예방해줍니다. 계란 속 시스테인이라는 성분은 알코올을 분해하는 데 도움을 줘서 피로회복에 효과적입니다. 셋째는 전해질 보충입니다. 술은 단순히 수분만 빼앗는 게 아니라 몸속 전해질 균형까지 무너뜨려요. 이럴 때 이온 음료나 미역국, 된장국 같은 따뜻한 국물을 섭취하면 부족해진 나트륨과 칼륨을 채워주어 몸을 안정시키고 두통을 완화할 수 있습니다. 단순한 해장이 아니라 회복을 위한 중요한 과정입니다. 넷째는 몸을 깨우는 방식입니다. 술이 아직 남아있는 상태에서 격한 운동을 하면 심장과 혈관에 큰 부담을 줍니다. 하지만 가볍게 걸으면서 땀을 조금 흘리거나 스트레칭을 하면 혈액 순환이 촉진되고 머리가 맑아지지요. 무리하지 않으면서도 몸을 깨우는 데는 산책과 가벼운 스트레칭만큼 좋은 방법이 없습니다. 마지막은 약을 피하는 겁니다. 술로 인한 두통 때문에 진통제를 찾는 경우가 많은데 전날 술이 덜 깬 상태에서 약을 복용하면 위장 출혈이나 간 손상 같은 심각한 문제가 생길 수 있습니다. 숙취은 약으로 해결하는 것이 아니라 물과 영양, 휴식으로 극복하는 것이 가장 안전한 방법입니다. 아침에 물을 충분히 마시고 소화 잘 되는 음식을 챙겨 먹고 필요하다면 20분 정도 낮잠을 자면서 간과 뇌를 쉬게 하는 것이 가장 좋은 해답입니다. 실제로 제가 만난 60대 남성 환자 한 분은 술만 마시면 늘 진통제를 복용하곤 했습니다. 결국 위장 출혈로 응급실에 실려왔는데 원인은 술과 약의 조합이었습니다. 이후 회복 루틴을 실천하면서 약을 완전히 끊으셨고 지금은 술을 드신 다음 날에도 큰 무리 없이 회복할 수 있을 정도로 몸 상태가 달라지셨습니다. 여러분, 술을 완전히 끊지 않더라도 이 회복 습관만 지켜도 다음 날의 몸 상태는 확실히 달라집니다. 단순해 보여도 꾸준히 실천하면 가장 큰 힘을 발휘합니다. 그리고 여기서 한 가지 더 중요한 질문이 남아있습니다. 술을 마시면서도 평생 건강을 지키려면 어디까지가 안전한 선일까요? 바로 다음 장에서 그 해답을 알려드리겠습니다. 여러분, 그렇다면 술을 완전히 끊지 않고도 건강을 지킬 수 있는 한계선은 어디까지일까요? 사실 이 질문은 환자분들이 진료실에서 가장 많이 묻는 내용 중 하나입니다. 선생님, 하루에 어느 정도까지는 괜찮습니까? 하구요. 하지만 이 답은 단순히 잔수로만 정할 수 있는 것이 아닙니다. 나이, 성별, 간 기능, 복용 중인 약 그리고 평소 생활 습관에 따라 크게 달라지기 때문입니다. 그럼에도 불구하고 연구와 임상 경험을 종합하면 참고할 만한 기준은 있습니다. 일반적으로 남성은 하루에 소주 한두 잔, 맥주 한 캔 정도, 여성은 그 절반 정도가 간에 무리를 주지 않는 양으로 알려져 있습니다. 하지만 여기서 중요한 건 매일 마시면 안 된다는 겁니다. 일주일에 이틀은 반드시 술을 쉬어야 간이 회복할 시간을 가질 수 있습니다. 운동에도 휴식일이 필요하듯 간에게도 쉬는 날이 꼭 필요하지요. 또 하나 중요한 점은 연속 음주를 피하는 겁니다. 오늘도 마시고 내일도 마시는 패턴이 가장 위험합니다. 술이 해롭다기보다 간에게 쉴 틈을 주지 않는 것이 더 큰 문제입니다. 그래서 저는 환자분들께 늘 말씀드립니다. 술을 마신 날에는 반드시 이틀 이상은 쉬어라. 그래야 몸이 다시 균형을 찾을 수 있다고요. 그리고 나이에 따라 안전선은 점점 낮아집니다. 젊었을 때는 괜찮았던 양이 60세, 70세가 넘어가면 고스란히 간과 심장, 뇌에 부담이 됩니다. 실제로 제가 진료했던 한 70대 어르신은 예전에는 소주 한 병은 거뜬했었는데 이제는 두 잔만 마셔도 다음 날이 너무 힘들다 하시더군요. 몸이 보내는 신호를 무시하지 않고 예전 기준을 고집하지 않는 것이 지혜로운 음주 습관입니다. 술을 마실 때는 양보다 더 중요한 것이 바로 속도와 습관입니다. 같은 두 잔이라도 천천히 물과 함께 음식을 곁들이며 마셨을 때와 단숨에 마셨을 때는 몸의 부담이 전혀 다릅니다. 또 같은 잔수라도 공복에 마시느냐, 충분히 식사 후에 마시느냐에 따라 결과가 달라지지요. 그러니 안전선은 단순히 몇 잔까지가 아니라 어떻게 마시는가에 달려있다라고 말씀드릴 수 있습니다. 여러분, 술은 결국 생활 속의 선택입니다. 어느 날은 술잔을 드는 대신 따뜻한 차 한 잔을 선택하는 지혜가 필요합니다. 술이 내 삶의 즐거움이 될 수는 있지만 내 몸과 가족의 행복을 위협하는 수준이 되어서는 안 되겠지요. 그리고 아마 궁금하실 겁니다. 그렇다면은 이미 오랫동안 술을 즐겨왔던 분들이라도 지금부터 실천할 수 있는 변화는 무엇일까? 또 실제로 습관을 바꿔서 건강을 되찾은 사례는 없을까? 바로 다음 장에서 그 생생한 이야기를 들려드리겠습니다. 여러분, 지금까지 말씀드린 원칙들을 들으면서 정말 효과가 있을까 하고 의심이 드실 수도 있습니다. 그런데 실제로 습관을 바꾼 환자분들의 사례를 보면 생각이 달라집니다. 제가 만난 70대 박영호 어르신은 평생 술을 친구처럼 여기셨습니다. 저녁 식탁에는 늘 술잔이 있었고 하루라도 술을 건너뛰면 잠이 안 온다고 하실 정도였습니다. 하지만 몇 년 전 건강 검진에서 간 수치가 크게 올라가면서 경고를 받으셨습니다. 그때 어르신은 처음으로 술 때문에 정말 큰일이 나겠다 하는 두려움을 느끼셨다고 합니다. 그 후 저는 어르신께 오늘 말씀드린 원칙들을 하나하나 알려드렸습니다. 공복에 술을 피하고 반드시 물과 함께 마시고 주량을 정해두는 단순한 습관들이었습니다. 처음엔 귀찮아하시고 이게 무슨 효과가 있겠어 하셨지만 놀라운 건 단 3주 만에 변화가 나타났다는 점입니다. 혈액 검사에서 간 수치가 정상으로 돌아왔고 무엇보다 아침에 상쾌하게 일어날 수 있게 되었다는 사실이었습니다. 어르신은 이렇게 말씀하셨습니다. 예전에는 술이 나를 위로해 준다고 생각했는데 지금은 내가 술을 다스리고 있는 것 같소. 덕분에 아침마다 손주와 함께 산책도 하고 밥맛도 돌아왔습니다. 또 다른 사례도 있습니다. 60대 여성 김선이 여사님은 남편과 함께 저녁마다 와인을 즐기셨습니다. 하지만 어느 순간부터 얼굴이 쉽게 붓고 심장이 두근거리는 증상이 나타났습니다. 저는 여사님께 일주일에 이틀은 반드시 술을 쉬는 날을 정하고 술을 마신 다음 날은 회복 루틴을 꼭 지키도록 조언드렸습니다. 그 결과 두 달 만에 혈압이 안정되고 피부 트러블도 줄어들어 거울 속 내 모습이 달라졌다고 하실 정도였습니다. 이 사례들이 보여주는 것은 분명합니다. 술을 완전히 끊지 않아도 됩니다. 단지 마시는 방식을 조금만 바꿔도 몸은 빠르게 반응합니다. 그리고 그 변화는 단순히 건강 수치에만 머무르지 않고 삶의 활력과 가족과의 관계까지 바꿔줍니다. 여러분, 혹시 지금도 술 때문에 몸이 힘들고 가족의 걱정을 듣고 계신가요? 그렇다면 오늘 말씀드린 작은 습관부터 하나씩 실천해보시길 권해드립니다. 변화는 생각보다 훨씬 빨리 찾아옵니다. 그리고 마지막으로 이렇게 바꾼 습관을 평생 지키려면 무엇이 필요할까요? 바로 다음 장에서 그 비밀을 말씀드리겠습니다. 8장. 여러분, 결국 술은 우리의 삶에서 완전히 없애기 어려운 친구일지도 모릅니다. 하지만 친구처럼 대하되 거리를 지켜야 오래도록 함께할 수 있습니다. 오늘 말씀드린 원칙들을 단순한 지식으로만 두지 마시고 생활 속 습관으로 만들어보시기 바랍니다. 우선 술을 마신 날에는 반드시 이틀 이상은 쉬어주는 것. 이건 간에게 주는 선물입니다. 그리고 술잔을 들 땐 반드시 물잔도 함께 두고 공복 상태는 절대 피하는 것. 이건 내 위와 간을 지키는 방패가 됩니다. 무엇보다도 술을 즐기기보다 술자리를 즐긴다는 마음으로 천천히 대화와 음식을 곁들여 보세요. 그렇게 하면 술은 적이 아니라 삶의 조미료가 될 수 있습니다. 제가 만난 많은 환자분들이 공통으로 하신 말씀이 있습니다. 처음에는 습관을 바꾸는 게 불편했지만 어느 순간부터 몸이 달라지는 걸 느끼고 나니 술보다 건강이 더 좋아졌다는 고백이었습니다. 그 말처럼 작은 변화가 모여 결국 평생을 바꿉니다. 그리고 꼭 기억해주세요. 술은 내 몸만의 문제가 아니라 가족의 행복과도 연결되어 있습니다. 나의 건강한 선택 하나가 배우자의 미소로 자녀의 안심으로 손주의 웃음으로 돌아옵니다. 그것이야말로 술을 지혜롭게 대하는 진짜 이유가 아닐까요? 이제 여러분께 부탁드립니다. 오늘 제가 전해드린 이야기 중에서 단 한 가지만이라도 실천해보세요. 그 작은 시작이 내일의 몸을 더 가볍게 하고 앞으로의 인생을 더 길고 활력 있게 만들어 줄 것입니다. 마지막으로 이 영상이 도움이 되셨다면 구독과 좋아요, 알림 설정까지 꼭 눌러주세요. 그리고 술 때문에 걱정하는 가족이나 친구에게도 함께 공유해주시길 바랍니다. 누군가의 인생을 바꾸는 전환점이 될 수 있기 때문입니다. 여러분, 술은 끊어야만 답이 아닙니다. 지혜롭게 다스릴 때 건강과 행복을 모두 지킬 수 있습니다. 앞으로도 제가 곁에서 함께 걸어가며 더 많은 건강 이야기를 나누겠습니다. 우리 모두의 노후가 더 길고 더 아름다워지기를 바랍니다. 감사합니다.    
"""
    
    # 파이프라인 실행
    final_state = run_fact_check_pipeline(sample_ad)