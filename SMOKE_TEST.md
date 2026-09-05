# AI 스모크 테스트 절차 (TASK-07)

수동 실행 스크립트 `test.py`, `test_legal.py`를 실제로 돌려볼 때 참고하는 절차 기록이다.
두 스크립트 모두 `sys.exit(0)`(PASS) / `sys.exit(1)`(FAIL)로 종료하므로 CI나 쉘에서
`echo $?`로 결과를 판정할 수 있다.

## 공통 사전조건

- `.env`(또는 실행 환경 변수)에 `GOOGLE_API_KEY`가 설정돼 있어야 한다.
- 비밀값(API 키, 쿠키 내용, PO 토큰)은 이 문서나 실행 로그에 값 그대로 남기지 않는다.
  실행 결과를 기록할 때는 "설정됨/비어있음", 상태 코드, 성공/실패만 남긴다.

## 1. `test.py` — 전체 파이프라인 (다운로드 → 전사 → 분석)

| 항목 | 내용 |
|---|---|
| 실행 | `python test.py <youtube_url>` |
| 추가 사전조건 | `YOUTUBE_COOKIE_PATH`(선택, 없으면 저장소 루트 `youtube_cookies.txt`), `MODEL_NAME0~9` 중 최소 1개 |
| 비용 발생 지점 | `get_transcript`가 자막을 못 찾으면 Gemini 전사(유료) 호출, `app.invoke`의 각 노드가 Gemini 호출(유료) |
| PASS 기준 | 영상 다운로드 성공 **and** 전사 텍스트가 비어있지 않음 **and** 최종 `report`가 비어있지 않음 |
| FAIL 기준 | 위 셋 중 하나라도 실패, 또는 예외 발생 |
| 대상 영상 선택 | 공개 영상으로 먼저 시도한다. PO 토큰 없이 실패하면(포맷 조회 불가) 로그인 전용/연령제한 등 PO 토큰이 필요한 영상으로 분류하고, 그 경우에만 `YOUTUBE_PO_TOKEN`을 채운 새 토큰으로 별도 검증한다. |

## 2. `test_legal.py` — 법률 판단 노드 단독

| 항목 | 내용 |
|---|---|
| 실행 | `python test_legal.py` (스크립트 내부에 샘플 광고 대본이 하드코딩돼 있음) |
| 추가 사전조건 | ChromaDB가 `CHROMA_HOST:CHROMA_PORT`에서 응답해야 함(`legal_documents` 컬렉션) |
| 비용 발생 지점 | `classify_domain` 1회, 최종 리포트 생성 1회 — Gemini 호출 2회(유료) |
| PASS 기준 | `legal` 결과가 존재하고 `legal_issue_score`가 0~1 범위이며 근거가 1건 이상 |
| FAIL 기준 | `legal` 결과 없음, 점수 범위 밖, 근거 0건, 또는 예외 발생(GOOGLE_API_KEY 누락·Chroma 연결 실패 등은 `nodes/legal.py`의 지연 초기화가 예외로 알려준다) |

## 3. 결과 기록 형식

각 실행마다 다음만 기록한다(비밀값 제외):

```
날짜/시각 | 스크립트 | 대상(영상 ID 또는 "샘플 대본") | PASS/FAIL | 소요 시간 | 비고
```

## 4. 이 문서의 범위가 아닌 것

- 실제 외부 연동의 최종 승인 판단은 Stage 4 검증(3.5 검증계획서 VAL-*)에서 별도로 수행한다.
- yt-dlp/YouTube 쪽 포맷 조회 성공 여부는 YouTube의 실시간 정책 변화에 따라 달라질 수 있어
  이 문서로 고정하지 않는다. 실패 시 먼저 `YOUTUBE_COOKIE_PATH`가 유효한 최신 쿠키를 가리키는지
  확인한다.
