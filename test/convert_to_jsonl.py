import json
import os
from pathlib import Path

def convert_json_to_jsonl(input_dir, output_file):
    """
    지정된 디렉토리의 JSON 파일을 하나의 JSONL 파일로 변환합니다.
    최대 800개의 파일만 처리합니다.
    
    Args:
        input_dir (str): JSON 파일이 있는 디렉토리 경로
        output_file (str): 출력할 JSONL 파일 경로
    """
    print(f"\n변환을 시작합니다...")
    print(f"입력 디렉토리: {input_dir}")
    print(f"출력 파일: {output_file}")
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 모든 JSON 파일 처리
    processed_files = 0
    total_conversations = 0
    max_files = 800  # 최대 파일 수 제한
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for json_file in Path(input_dir).glob("*.json"):
            if processed_files >= max_files:
                break
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 각 대화 처리
                for info in data['info']:
                    lines = info['annotations']['lines']
                    conversation = []
                    
                    # 대화 내용 수집
                    for line in lines:
                        # 발화자 정보와 내용을 결합
                        speaker_info = f"{line['speaker']['sex']} {line['speaker']['age']}"
                        content = f"{speaker_info}: {line['norm_text']}"
                        conversation.append(content)
                    
                    # 대화를 training format으로 변환
                    if len(conversation) >= 2:
                        for i in range(0, len(conversation)-1, 2):
                            if i+1 < len(conversation):
                                # 프롬프트와 응답 형식으로 변환
                                prompt = conversation[i]
                                completion = conversation[i+1]
                                
                                # JSONL 형식으로 저장
                                jsonl_item = {
                                    "messages": [
                                        {"role": "user", "content": prompt},
                                        {"role": "assistant", "content": completion}
                                    ]
                                }
                                outfile.write(json.dumps(jsonl_item, ensure_ascii=False) + '\n')
                                total_conversations += 1
                
                processed_files += 1
                print(f"처리된 파일: {json_file.name} ({processed_files}/{max_files})")
                
            except Exception as e:
                print(f"파일 처리 중 오류 발생 ({json_file.name}): {str(e)}")
    
    # 결과 출력
    print(f"\n변환 완료!")
    print(f"처리된 파일 수: {processed_files}")
    print(f"생성된 대화 쌍: {total_conversations}")
    
    # 파일 크기 확인
    file_size = os.path.getsize(output_file)
    print(f"생성된 파일 크기: {file_size / 1024:.2f} KB")

def main():
    # 입력 디렉토리와 출력 파일 경로 설정
    input_dir = "/Users/yuseunghwan/Downloads/020.주제별 텍스트 일상 대화 데이터/01.데이터/1.Training/라벨링데이터/TL_03. INSTAGRAM"
    output_file = "/Users/yuseunghwan/workspace/dev-project/toy-project/chatpression/instagram_training_data.jsonl"
    
    # 변환 실행
    convert_json_to_jsonl(input_dir, output_file)

if __name__ == "__main__":
    main() 