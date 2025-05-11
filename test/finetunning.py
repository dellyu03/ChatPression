import os
import json
import openai
import time
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

def check_api_key():
    """API 키가 올바르게 설정되어 있는지 확인합니다."""
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    print("API 키 확인 완료")

def list_fine_tuning_jobs():
    """현재 진행 중인 파인튜닝 작업 목록을 확인합니다."""
    print("\n현재 파인튜닝 작업 상태 확인 중...")
    try:
        jobs = openai.fine_tuning.jobs.list()
        if not jobs.data:
            print("진행 중인 파인튜닝 작업이 없습니다.")
            return
        
        print("\n현재 파인튜닝 작업 목록:")
        for job in jobs.data:
            print(f"\n작업 ID: {job.id}")
            print(f"상태: {job.status}")
            print(f"모델: {job.model}")
            print(f"생성 시간: {datetime.fromtimestamp(job.created_at)}")
            if job.fine_tuned_model:
                print(f"파인튜닝된 모델: {job.fine_tuned_model}")
            if job.error:
                print(f"에러: {job.error}")
    except Exception as e:
        print(f"작업 목록 조회 중 오류 발생: {str(e)}")

def cancel_fine_tuning_job(job_id):
    """파인튜닝 작업을 취소합니다."""
    try:
        response = openai.fine_tuning.jobs.cancel(job_id)
        print(f"\n작업 {job_id}가 취소되었습니다.")
        print(f"상태: {response.status}")
    except Exception as e:
        print(f"작업 취소 중 오류 발생: {str(e)}")

def validate_training_data(training_data):
    """학습 데이터의 품질을 검증합니다."""
    print("\n학습 데이터 검증 중...")
    
    if not training_data:
        raise ValueError("학습 데이터가 비어있습니다.")
    
    valid_data = []
    invalid_data = []
    
    for item in training_data:
        try:
            # 메시지 형식 검증
            if not isinstance(item, dict) or 'messages' not in item:
                invalid_data.append(("메시지 형식 오류", item))
                continue
                
            messages = item['messages']
            if not isinstance(messages, list) or len(messages) != 2:
                invalid_data.append(("메시지 개수 오류", item))
                continue
                
            # user와 assistant 메시지 검증
            if messages[0]['role'] != 'user' or messages[1]['role'] != 'assistant':
                invalid_data.append(("역할 지정 오류", item))
                continue
                
            # 메시지 내용 검증
            if not messages[0]['content'] or not messages[1]['content']:
                invalid_data.append(("빈 메시지 내용", item))
                continue
                
            # 메시지 길이 검증
            if len(messages[0]['content']) < 5 or len(messages[1]['content']) < 5:
                invalid_data.append(("메시지가 너무 짧음", item))
                continue
                
            valid_data.append(item)
            
        except Exception as e:
            invalid_data.append((str(e), item))
    
    print(f"\n검증 결과:")
    print(f"- 유효한 데이터: {len(valid_data)}개")
    print(f"- 유효하지 않은 데이터: {len(invalid_data)}개")
    
    if invalid_data:
        print("\n유효하지 않은 데이터 샘플:")
        for reason, item in invalid_data[:5]:  # 처음 5개만 출력
            print(f"- 이유: {reason}")
            print(f"  데이터: {json.dumps(item, ensure_ascii=False)}")
    
    return valid_data

def convert_to_training_format(data_dir):
    """인스타그램 대화 데이터를 OpenAI 파인튜닝 형식으로 변환합니다."""
    training_data = []
    processed_files = 0
    
    print("\n데이터 변환을 시작합니다...")
    
    # 모든 JSON 파일 처리
    for json_file in Path(data_dir).glob("*.json"):
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
                            
                            training_data.append({
                                "messages": [
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": completion}
                                ]
                            })
            
            processed_files += 1
            print(f"처리된 파일: {json_file.name}")
            
        except Exception as e:
            print(f"파일 처리 중 오류 발생 ({json_file.name}): {str(e)}")
    
    print(f"\n총 {processed_files}개의 파일이 처리되었습니다.")
    print(f"생성된 대화 쌍: {len(training_data)}개")
    return training_data

def save_training_data(training_data, output_file):
    """학습 데이터를 JSONL 파일로 저장합니다."""
    print(f"\n학습 데이터를 {output_file}에 저장합니다...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 파일 크기 확인
    file_size = os.path.getsize(output_file)
    print(f"저장된 파일 크기: {file_size / 1024:.2f} KB")
    
    return file_size

def upload_file(file_path):
    """파일을 OpenAI에 업로드합니다."""
    print(f"\n파일 업로드 중: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            response = openai.files.create(
                file=f,
                purpose='fine-tune'
            )
        print(f"파일 업로드 완료. 파일 ID: {response.id}")
        return response.id
    except Exception as e:
        print(f"파일 업로드 중 오류 발생: {str(e)}")
        raise

def create_fine_tune(file_id):
    """파인튜닝 작업을 생성합니다."""
    print("\n파인튜닝 작업 생성 중...")
    try:
        # 현재 작업 상태 확인
        list_fine_tuning_jobs()
        
        # 사용자에게 계속 진행할지 확인
        response = input("\n파인튜닝을 계속 진행하시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("파인튜닝을 취소합니다.")
            return None
        
        response = openai.fine_tuning.jobs.create(
            training_file=file_id,
            model="gpt-3.5-turbo"
        )
        print(f"파인튜닝 작업 생성 완료. 작업 ID: {response.id}")
        return response.id
    except Exception as e:
        print(f"파인튜닝 작업 생성 중 오류 발생: {str(e)}")
        if "rate_limit_exceeded" in str(e):
            print("\n현재 파인튜닝 작업이 너무 많습니다. 다음 중 하나를 선택하세요:")
            print("1. 기존 작업이 완료될 때까지 기다리기")
            print("2. 기존 작업 중 일부를 취소하기")
            choice = input("선택 (1/2): ")
            if choice == "2":
                job_id = input("취소할 작업 ID를 입력하세요: ")
                cancel_fine_tuning_job(job_id)
        raise

def monitor_fine_tune(job_id):
    """파인튜닝 작업의 진행 상황을 모니터링합니다."""
    if not job_id:
        return None
        
    print("\n파인튜닝 진행 상황 모니터링을 시작합니다...")
    print("파인튜닝은 보통 20-30분 정도 소요됩니다.")
    
    while True:
        try:
            job = openai.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            
            # 현재 시간 출력
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if status == "succeeded":
                print(f"\n[{current_time}] 파인튜닝이 성공적으로 완료되었습니다!")
                print(f"파인튜닝된 모델: {job.fine_tuned_model}")
                return job.fine_tuned_model
            elif status == "failed":
                print(f"\n[{current_time}] 파인튜닝이 실패했습니다.")
                print(f"에러 메시지: {job.error}")
                return None
            else:
                print(f"\n[{current_time}] 현재 상태: {status}")
                if hasattr(job, 'trained_tokens'):
                    print(f"학습된 토큰 수: {job.trained_tokens}")
        
        except Exception as e:
            print(f"상태 확인 중 오류 발생: {str(e)}")
        
        # 60초 대기
        time.sleep(60)

def test_fine_tuned_model(model_id, test_prompt):
    """파인튜닝된 모델을 테스트합니다."""
    if not model_id:
        print("테스트할 모델이 없습니다.")
        return
    
    print(f"\n파인튜닝된 모델 테스트 중...")
    print(f"테스트 프롬프트: {test_prompt}")
    
    try:
        response = openai.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": test_prompt}],
            max_tokens=300,
            temperature=0.7
        )
        print("\n모델 응답:")
        print(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"모델 테스트 중 오류 발생: {str(e)}")

def main():
    try:
        # API 키 확인
        check_api_key()
        
        # 데이터 변환
        data_dir = "/Users/yuseunghwan/Downloads/020.주제별 텍스트 일상 대화 데이터/01.데이터/1.Training/라벨링데이터/TL_03. INSTAGRAM"
        training_data = convert_to_training_format(data_dir)
        
        if not training_data:
            print("변환된 데이터가 없습니다. 데이터 디렉토리를 확인해주세요.")
            return
        
        # 데이터 저장
        output_file = "instagram_training_data.jsonl"
        file_size = save_training_data(training_data, output_file)
        
        if file_size < 1024:  # 1KB 미만
            print("경고: 파일이 너무 작습니다. 더 많은 데이터가 필요할 수 있습니다.")
        
        # 파일 업로드 및 파인튜닝 시작
        file_id = upload_file(output_file)
        job_id = create_fine_tune(file_id)
        
        # 파인튜닝 모니터링
        fine_tuned_model = monitor_fine_tune(job_id)
        
        # 테스트 프롬프트로 모델 테스트
        if fine_tuned_model:
            test_prompt = "여성 30대: 오늘 점심 뭐 먹을까?"
            test_fine_tuned_model(fine_tuned_model, test_prompt)
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        print("프로그램을 종료합니다.")

if __name__ == "__main__":
    main()