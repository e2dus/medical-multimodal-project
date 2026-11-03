# ==========================================
# Dockerfile: ECG 1-Lead -> 12-Lead Reconstruction
# ==========================================
# 서버 환경:
#   - GPU: NVIDIA TITAN X (Pascal) x8 (12GB each)
#   - Driver: 510.60.02
#   - CUDA 지원: 11.6 (드라이버 기준 최대)
#   - OS: Ubuntu 16.04
#
# PyTorch 1.13.1 + CUDA 11.6 사용 (드라이버 호환)
# TF 2.10 (마지막 CUDA 11.6 지원 버전) — EKGAN baseline 용
# ==========================================

FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /workspace

# 의존성 먼저 복사 (Docker 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# TensorFlow 2.10 (GPU, CUDA 11.6 호환) — EKGAN baseline
# PyTorch 1.13.1 cudnn8과 호환
RUN pip install --no-cache-dir tensorflow==2.10.0

# 프로젝트 코드 복사
COPY ecg_models/models/ ./ecg_models/models/
COPY ecg_1to12/ ./ecg_1to12/
COPY PAPERS.md ./PAPERS.md
COPY run_all.sh ./run_all.sh
COPY inference.py ./inference.py
RUN chmod +x run_all.sh

# ★ PRIME-ECG 전용: cascade_recon.py 복사
#    (사용자가 프로젝트 루트에 cascade_recon.py를 먼저 복사해둘 것)
COPY cascade_recon.py ./cascade_recon.py

# 데이터/출력/추론결과 디렉토리 (마운트 포인트)
RUN mkdir -p /workspace/data/ptb-xl
RUN mkdir -p /workspace/outputs
RUN mkdir -p /workspace/results

# 환경 변수
ENV PYTHONPATH=/workspace:/workspace/ecg_models/models
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# 기본: PRIME-ECG 실험 실행
# docker run --rm --gpus all -v /data/ptb-xl:/workspace/data/ptb-xl ... \
#   python -m ecg_models.models.prime_ecg_full_experiments --data_dir /workspace/data/ptb-xl
CMD ["python", "-m", "ecg_models.models.prime_ecg_full_experiments", "--help"]
