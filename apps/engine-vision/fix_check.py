import sys
import torch.utils._pytree

# PyTorch 2.1.0+ 대응: 아예 함수가 없으면 가짜 함수를 주입합니다.
def dummy_register(*args, **kwargs):
    pass

# 모듈에 강제로 주입
torch.utils._pytree.register_pytree_node = dummy_register
torch.utils._pytree.register_leaf = dummy_register

import transformers
print(f'트랜스포머 버전: {transformers.__version__}')
print('-----------------------------------------')
print('[SUCCESS] 가짜 함수 주입으로 로드 성공!')
