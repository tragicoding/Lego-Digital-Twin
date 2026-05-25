# Mock Character Assets

이 폴더에 **리깅된 FBX** 와 **텍스쳐 GLB** 파일 쌍을 넣으세요.

## 파일 네이밍 규칙

| 파일 | 역할 |
|---|---|
| `npc_N_rigged.fbx` | 리깅된 캐릭터 FBX (텍스쳐 없음) |
| `npc_N_texture.glb` | PBR 텍스쳐 소스 GLB |

> 반드시 같은 `npc_N` 접두사를 사용해야 합니다.

---

## 자동 텍스쳐 적용 (CharacterTextureApplier)

두 파일을 이 폴더에 복사하면 Unity가 임포트하면서 **자동으로** 텍스쳐를 적용합니다.

### 처리 흐름
1. FBX 또는 GLB 임포트 감지
2. 페어 파일 존재 확인
3. GLB에서 Texture2D (BaseColor / Normal / Metallic) 추출
4. URP Lit 머티리얼(`.mat`) 생성 → `Materials/npc_N.mat`
5. FBX 머티리얼을 해당 `.mat` 으로 리맵 후 재임포트

### 수동 실행
Unity 메뉴: **Tools > MINIVERSE > Apply Character Textures**  
(GLB가 이미 임포트되어 있어야 동작합니다.)

---

## Inspector 연결

텍스쳐가 적용된 FBX로 Prefab을 만든 뒤:
- `GeneratedCharacterSpawner.mockCharacterPrefab` 필드에 드래그

---

## 현재 파일 목록

| 파일 | 상태 |
|---|---|
| `npc_3_rigged.fbx` | ✅ 테스트용 FBX |
| `npc_3_texture.glb` | ✅ 테스트용 텍스쳐 GLB |

---

## 서버 모드 전환 시

서버에서 FBX URL + 텍스쳐 GLB URL 이 내려오면 `GeneratedCharacterSpawner` 가  
런타임에 다운로드 + 적용합니다. (TriLib + glTFast 구현 필요)
