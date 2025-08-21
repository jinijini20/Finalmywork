from openai import OpenAI
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_qa_and_sources(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """주어진 구글의 People API에 대한 문서를 바탕으로 유용한 질문-답변과 해당 답변의 출처 URL을 찾아주세요.

**중요한 제약사항:**
- 문서에 명시된 내용만을 기반으로 질문과 답변을 작성하세요
- 문서에 없는 내용이나 추측, 일반적인 지식을 추가하지 마세요
- 답변은 반드시 문서 내용을 직접 참조해야 합니다
- 확실하지 않은 내용은 포함하지 마세요

**우선적으로 다룰 주제:**
1. 코드 예시와 구현 방법
2. API 사용법과 파라미터 설명
3. 오류 해결 방법과 문제 해결책
4. 설정 방법과 구성 옵션
5. 실제 사용 사례와 예제

**구체적인 서비스 사용법:**
- 문서에서 다루는 `People API`의 각 서비스나 기능, 예시 코드의 언어 등을 명시적으로 질문과 답변에 포함시켜 주세요. 예를 들어, `Geocoding`, `Distance Matrix API`, `Street View` 등 Map의 **구체적인 기능**에 대해 질문을 생성합니다.
- 문서에 나오는 **서비스별 사용법**이나 **구성 옵션**도 질문과 답변에 녹여내야 합니다.

**생성 규칙:**
- 문서에서 위 주제에 해당하는 내용이 충분히 있을 때만 질문-답변을 생성하세요
- 내용이 부족하거나 추상적인 설명만 있다면 "생성할 수 없음"이라고 응답하세요
- 최대 5개의 질문-답변을 생성하세요
- 각 질문-답변은 실용적이고 구체적이어야 합니다

**출처 URL 찾기 규칙:**
1. 문서 맨 위에 있는 페이지 URL (보통 https://developers.google.com/people 와 같이 시작)을 기본으로 포함
2. 답변 내용과 직접 관련된 특정 섹션이나 기능의 URL이 문서 내에 별도로 명시되어 있다면 추가로 포함
3. 내부 링크나 참조 링크가 답변과 관련이 있다면 포함
4. 문서에 실제로 존재하는 URL만 사용하세요

**응답 형식:**
질문1: [문서 내용 기반의 구체적인 질문]
답변1: [문서에 명시된 내용만으로 작성한 답변]
출처1: [문서 내에 실제로 존재하는 URL1, URL2, ...]

질문2: [두 번째 질문]
답변2: [두 번째 답변]
출처2: [URL1, URL2, ...]

(최대 5개까지)

만약 적절한 코드 예시나 사용법, 오류 해결법이 문서에 충분히 없다면 "생성할 수 없음"이라고만 응답하세요."""
                },
                {
                    "role": "user",
                    "content": f"다음 구글의 people에 대한 API 문서를 정확히 분석해서, 코드 예시, 사용법, 오류 해결법 등 실용적인 내용을 기반으로 최대 5개의 질문-답변과 해당 출처 URL들을 찾아주세요. 적절한 내용이 없다면 '생성할 수 없음'이라고 응답하세요:\n\n{text}"
                }
            ],
            max_tokens=2000,
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"질문-답변 및 출처 생성 중 오류 발생: {e}")
        return None


def parse_single_qa_block(block):
    """단일 질문-답변-출처 블록 파싱"""
    try:
        qa_dict = {}
        lines = block.strip().split('\n')

        current_section = None
        question_lines = []
        answer_lines = []
        source_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 질문 시작 패턴
            if re.match(r'^질문\d*:', line):
                current_section = 'question'
                question_content = re.sub(r'^질문\d*:', '', line).strip()
                if question_content:
                    question_lines.append(question_content)

            # 답변 시작 패턴
            elif re.match(r'^답변\d*:', line):
                current_section = 'answer'
                answer_content = re.sub(r'^답변\d*:', '', line).strip()
                if answer_content:
                    answer_lines.append(answer_content)

            # 출처 시작 패턴
            elif re.match(r'^출처\d*:', line):
                current_section = 'sources'
                source_content = re.sub(r'^출처\d*:', '', line).strip()
                if source_content:
                    source_lines.append(source_content)

            # 연속되는 내용 라인
            else:
                if current_section == 'question':
                    question_lines.append(line)
                elif current_section == 'answer':
                    answer_lines.append(line)
                elif current_section == 'sources':
                    source_lines.append(line)

        # 각 섹션 조합
        if question_lines:
            qa_dict['question'] = ' '.join(question_lines).strip()

        if answer_lines:
            qa_dict['answer'] = ' '.join(answer_lines).strip()

        # URL은 그대로 사용 (중복 파싱 제거)
        if source_lines:
            # AI가 이미 주는 URL을 그대로 분리
            urls = []
            for s in source_lines:
                urls.extend([u.strip() for u in s.split() if u.startswith("http")])
            qa_dict['sources'] = urls if urls else ["출처를 찾을 수 없음"]
        else:
            qa_dict['sources'] = ["출처를 찾을 수 없음"]

        return qa_dict

    except Exception as e:
        print(f"단일 QA 블록 파싱 중 오류: {e}")
        return {}


# AI 응답을 파싱하는 함수 (개선된 버전)
def parse_qa_and_sources(ai_response):
    """개선된 질문-답변-출처 파싱 함수"""
    try:
        # "생성할 수 없음" 응답 체크
        if "생성할 수 없음" in ai_response:
            print("  -> 적절한 내용이 없어서 질문-답변을 생성하지 않음")
            return []

        qa_pairs = []

        # 전체 텍스트를 질문 단위로 분할
        # 질문1:, 질문2: 등의 패턴으로 분할
        question_blocks = re.split(r'\n(?=질문\d*:)', ai_response.strip())

        for block in question_blocks:
            if not block.strip():
                continue

            # 각 블록에서 질문, 답변, 출처 추출
            qa_dict = parse_single_qa_block(block)
            if qa_dict and qa_dict.get('question') and qa_dict.get('answer'):
                # 데이터 정리
                cleaned_qa = {
                    'question': qa_dict['question'].strip(),
                    'answer': qa_dict['answer'].strip(),
                    'sources': qa_dict.get('sources', ["출처를 찾을 수 없음"])
                }
                qa_pairs.append(cleaned_qa)

        print(f"  -> 파싱된 QA 쌍 개수: {len(qa_pairs)}")
        return qa_pairs

    except Exception as e:
        print(f"응답 파싱 중 오류: {e}")
        return []


# 파일이 저장된 최상위 디렉토리 경로
root_dir = './people_docs'  # 원하는 폴더명으로 변경

def process_files_and_generate_jsonl():
    jsonl_data = []

    # 모든 폴더와 파일 순회 (os.walk 사용)
    all_files = []
    
    # os.walk를 사용하여 모든 폴더 및 파일 순회
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.txt'):
                all_files.append(os.path.join(root, filename))
    
    total_files = len(all_files)  # 총 파일 개수
    print(f"총 {total_files}개의 파일이 처리됩니다.")

    # 파일 처리 대기열 상태를 확인하며 순차적으로 처리
    for index, file_path in enumerate(all_files):
        print(f"처리 중: {index + 1}/{total_files} - {file_path}")  # 현재 처리 중인 파일 번호 출력

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # AI가 질문-답변과 출처를 한 번에 생성
        ai_response = generate_qa_and_sources(text)
        if ai_response:
            # AI 응답 파싱 (여러 개의 질문-답변 쌍)
            qa_pairs = parse_qa_and_sources(ai_response)

            if qa_pairs:
                print(f"  -> {len(qa_pairs)}개의 질문-답변 쌍 생성됨")
                for i, qa in enumerate(qa_pairs, 1):
                    print(f"    질문{i}: {qa['question'][:50]}...")
                    print(f"    출처{i}: {qa['sources']}")

                    # 메타데이터 설정
                    metadata = {
                        "question": qa['question'],
                        "answer": qa['answer'],
                        "sources": qa['sources'],
                        "tags": "people",
                        "last_verified": "2025-08-19",
                        "source_file": filename
                    }
                    jsonl_data.append(metadata)
            else:
                print(f"  -> 적절한 내용이 없어서 건너뛰기: {file_path}")
        else:
            print(f"  -> AI 응답 실패: {file_path}")

    # JSONL 파일로 저장
    jsonl_filename = 'generated_qa_people.jsonl'
    with open(jsonl_filename, 'w', encoding='utf-8') as jsonl_file:
        for item in jsonl_data:
            jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n완료! 총 {len(jsonl_data)}개의 질문-답변 쌍이 {jsonl_filename}에 저장되었습니다.")

# 실행
if __name__ == "__main__":
    process_files_and_generate_jsonl()
