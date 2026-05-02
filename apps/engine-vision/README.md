분류,기술 규격 (Syntax/Technology),비고
OS / Environment,Ubuntu 24.04 (WSL2) / Python 3.9,최신 리눅스 커널 기반
Deep Learning,PyTorch 2.1.0 + cu118 / Torchvision 0.16.0,40시리즈 최적화 조합
Compiler,GCC-11 / G++-11,CUDA 11.8 호환 컴파일러 고정
Accelerators,Tiny-cuda-nn (Direct Build) / Nerfacc 0.3.5,수동 빌드로 최대 성능 확보
Hot-fix,torch.utils._pytree Mocking,버전 불일치 해결 핵심 코드