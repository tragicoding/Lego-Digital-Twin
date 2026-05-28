**1. 시나리오 이해하기**

(1) NPC 가이드의 등장과 환영: 관람객이 처음 입장하면 플레이어의 왼쪽에서 대기하고 있던 NPC가 나타나 환영 인사를 건넵니다. 이 NPC는 일반적인 캐릭터가 아니라, 관람객이 직접 만든 레고 캐릭터(예: '로라')입니다.  

*"안녕하세요, MINIVERSE에 온 걸 환영해요 ! "*

*"저는 당신이 만든 캐릭터 000 이라고 해요. 만나서 반가워요 ! "*



(2) 월드 소개: NPC가 해리포터, 디즈니, 미래도시, 동심 월드 등 4개의 구역과 어트랙션으로 구성된 MINIVERSE 월드의 전반적인 배경을 설명해 줍니다.  

*"먼저 World를 소개할게요."* 

*"이곳은 4개의 구역으로 나눠져 있어요."*

*"해리포터, 디즈니, 미래도시, 동심 월드까지"* 

*"다양한 구역에는 즐길 수 있는 어트랙션도 있답니다."* 



(3) 이동하며: 

*"아참 !  World를 즐기기 전에, 광장 구역으로 가볼까요?"*

*"광장에서는 직접 만든 창작물들을 볼 수 있어요 ! "*



(4) 광장으로의 이동 및 창작물 확인: 안내에 따라 광장에 전시된 자신의 창작물로 이동합니다. 그곳에서 관람객 본인이 만든 레고 캐릭터(npc와 동일한 캐릭터)와 오브제를 확인하고, 캐릭터의 모션 등을 설정 및 확인하는 단계를 거칩니다.  

*"저기 다양한 오브제들이 보이네요, 가서 확인해볼까요?"*

*-만든 창작물 확인*

\-프롬프트 입력 모션 체험

\-시그니처 동작, 말풍선 설정



(5) 투표 설명: 본격적인 탐험에 앞서 다른 관람객들의 창작물에 '하트'를 누르는 투표 방법을 안내받습니다. 창작물 근처에 다가가 표시된 상호작용(누르기)을 통해 투표할 수 있으며 , 가장 많은 투표를 받은 1위 창작물 위에는 랭킹을 알리는 별 표시가 뜬다는 규칙을 배웁니다. 

*"들어가기 전, 광장에서는 마음에 드는 오브제에 하트를 누를 수 있어요."*

*"가장 하트가 많은 창작물은 별 표시가 있답니다 !"*



*"이제 당신의 MINIVERSE를 즐겨보세요 ! "*



\-자유 모드 전환: 모든 안내가 끝나면 '가이드 모드'가 종료되고 '자유 모드'로 전환됩니다. 투표기능이 가능하고, 동작프롬프트 입력기능이 가능합니다.



**2. 전체흐름(Mode 전환) 파악하기**

\[가이드 모드] 현재 관람객 캐릭터가 월드 안내

&#x09;- 입구 등장해서 가이드가 인사

&#x09;- 광장으로 이동하여 내 창작물(캐릭터+오브젝트)이 있는곳으로 안내

&#x09;- 프롬프트 입력 모션 체험

&#x09;- 시그니처 동작, 말풍선 설정

&#x20;     ↓ OnGuideFinished 이벤트

\[자유 모드 / Plaza]

&#x20; - 이전 관람객들의 캐릭터 + 오브제가 광장에 나열됨.

&#x20; - 다가가면 좋아요 버튼 등장

&#x20; - 좋아요 많은 창작물에 별(★) 표시

&#x20; - 좋아요 수는 WebSocket으로 실시간 갱신



Session 이란?

→ 관람객이 앱에서 “시작하기” 버튼을 누르는 순간, 그 관람객에 대한 data가 만들어지는데, 이를 “Session 생성” 이라고 함.

\- Session이 생기면 위에 있는 Json 데이터가 생성되고 DataBase에 추가됨.



**3. Script 디렉토리 구조 파악하기: 각 디렉토리/파일별 역할**

Scripts/

├── Character/

│   ├── CharacterAnimationController.cs  — 애니메이터 제어 (walk/idle/motion)

│   ├── GeneratedCharacterSpawner.cs     — 캐릭터 스폰 (Mock Prefab / Server GLB)

│   ├── GuideNPCController.cs            — 가이드 시나리오 진행 + 이동

│   ├── MixamoMotionLibrary.cs           — ScriptableObject (모션 클립 보관)

│   ├── MotionPromptParser.cs            — 한/영 키워드 → MotionType 변환

│   ├── MotionType.cs                    — 모션 enum (Dance, Run, Jump...)

│   ├── PlacedCharacterController.cs     — 배치 캐릭터 모션 재생

│

├── Core/

│   └── ServerConfig.cs                  — 서버 IP/포트 설정

│

├── Data/

│   └── SessionData.cs                   — 모든 데이터 클래스 정의

│

├── Managers/

│   ├── DataSourceMode.cs                — enum (Mock / Server)

│   └── SessionManager.cs               — 세션 로드 진입점

│

├── Mock/

│   └── MockSessionLoader.cs             — mock\_session.json 로드

│

├── Network/

│   ├── ApiClient.cs                     — REST API 호출 (GET/POST)

│   └── WebSocketManager.cs             — WS 실시간 이벤트 수신

│

├── Object/

│   └── GeneratedObjectSpawner.cs        — 오브제 스폰

│

├── Plaza/

│   ├── LikeSystem.cs                    — 근접 감지 + 좋아요 POST

│   ├── PlazaManager.cs                  — 광장 전체 관리 (스폰 + 실시간 갱신)

│   └── PlazaSessionView.cs             — 광장 내 세션 하나의 UI (좋아요수·말풍선·별)

│

└── World/

&#x20;   ├── BalloonTour.cs                   — 풍선 투어 연출

&#x20;   └── FastSkyDayNight.cs              — 낮/밤 하늘 전환



**4. Json 데이터 → 클래스 필드 파악**

**: Server에서 날라오는 Json Data 형태**

{

&#x20; "session\_id": "mock\_001",

&#x20; "character\_npc\_name": "몽글이",

&#x20; "object\_name": "하늘탑",

&#x20; "bubble\_text": "MINIVERSE에 온 걸 환영해!",

&#x20; "likes": 0,

&#x20; "ready\_for\_unity": true,

&#x20; "assets": {

&#x20;   "character": {

&#x20;     "asset\_id": "mock\_character\_001",

&#x20;     "model\_url": "",

&#x20;     "texture\_url": "",

&#x20;     "role": "guide\_npc",

&#x20;     "npc\_name": "몽글이"

&#x20;   },

&#x20;   "object": {

&#x20;     "asset\_id": "mock\_object\_001",

&#x20;     "model\_url": "",

&#x20;     "role": "static\_object",

&#x20;     "object\_name": "하늘탑"

&#x20;   }

&#x20; }

}



**: Json → Unity 에서 사용할 C# Class로의 변환(mapping)**

💡Script/Data/SessionData.cs → 클래스 정의



\## SessionData

\- 방금 서버로부터 응답받은 “현재 session”에 대한 모든 정보를 일단 담는 곳

public class SessionData

&#x20;   {

&#x20;       public string session\_id;

&#x20;       public string character\_npc\_name;

&#x20;       public string object\_name;

&#x20;       public string bubble\_text;

&#x20;       public int    likes;

&#x20;       

&#x20;       public bool   ready\_for\_unity;

&#x20;       

&#x20;       public SessionAssets assets; // -> SessionAssets 클래스

&#x20;   }

💡

\- 이후에 SessionManager가 로드해서 GuideNPCController.Initialize(Session)에 통째로 넘길거임.

\- SessionManager는 Json→객체로 변환해주고, Unity 전체에게 이 객체를 알림.

\- GuideNPCController.Initialize(Session) → Unity에 올라간 캐릭터가 초기화 된다.



\## SessionAssets

&#x20;  public class SessionAssets

&#x20;   {

&#x20;       public CharacterAssetData character;



&#x20;       // "object"는 C# 예약어 → @ 접두어 사용, JSON 키 "object"와 자동 매핑

&#x20;       public ObjectAssetData @object;

&#x20;   }

&#x20;   //둘다 아래에서 클래스 정의.



\## CharacterAssetData

&#x20;public class CharacterAssetData

&#x20;   {

&#x20;       public string asset\_id;

&#x20;       public string model\_url;      // FBX URL (서버) or "" (Mock → Inspector prefab 사용)

&#x20;       public string texture\_url;    // pbr\_model GLB URL (텍스쳐 소스, glTFast로 로드)

&#x20;       public string role;           // guide\_npc

&#x20;       public string npc\_name;

&#x20;   }

💡

\- asset\_id

&#x20;   - Session ID (PK값)

&#x20;   - Asset 구별할때 사용. (mock\_001, s\_ba001…)

\- model\_url

&#x20;   - 캐릭터의 FBX 파일경로 (개발중에 경로 바뀔 수도 있음)

\- texture\_url

&#x20;   - 캐릭터의 GLB 파일경로 (FBX에 텍스쳐를 입히기 위한 존재)

\- role

`role == "guide\_npc"    → 가이드 모드에서 안내 NPC로 스폰`

&#x20;   - Guide Mode에서 캐릭터는 두개가 띄워 질거임. 하나는 가이드, 하나는 배치.



\## ObjectAssetData

&#x20;   public class ObjectAssetData

&#x20;   {

&#x20;       public string asset\_id;

&#x20;       public string model\_url;      // GLB URL (서버) or "" (Mock → Inspector prefab 사용)

&#x20;       public string role;           // static\_object

&#x20;       public string object\_name;

&#x20;   }



\## PlazaResponse

&#x20;   public class PlazaResponse

&#x20;   {

&#x20;       public List<PlazaSessionData> sessions;

&#x20;       public string top\_session\_id;           // 현재 좋아요 1위 세션 ID

&#x20;   }



예시

//list 정렬 순서는 등록순서(세션 들어오는 순서대로)



sessions = \[

&#x20;   { session\_id: "session\_001", likes: 5 },   ← sessions\[0]

&#x20;   { session\_id: "session\_002", likes: 12 },  ← sessions\[1]  ★ 1위

&#x20;   { session\_id: "session\_003", likes: 3 },   ← sessions\[2]

]



//좋아요 1위 

top\_session\_id = "session\_002"   ← sessions\[1] 이 1위

//1위 로직은 PlazaManazer.cs에 구현되어 있음.



plaza.sessions.Count      // 관람객 수

plaza.sessions\[0]         // 첫 번째 관람객 (PlazaSessionData)

plaza.sessions\[0].likes   // 첫 번째 관람객 좋아요 수

top\_session\_id — 현재 좋아요 1위 세션 ID



// PlazaManager에서 별 표시 결정할 때 사용

plaza.top\_session\_id   // "session\_001"



\# PlazaSessionData

`PlazaResponse.sessions`  의 리스트 요소 하나에 해당하는 클래스.

&#x20;public class PlazaSessionData

&#x20;   {

&#x20;       public string session\_id;

&#x20;       public string character\_npc\_name;

&#x20;       public string bubble\_text;

&#x20;       public int    likes;

&#x20;       public bool   is\_top\_liked;             // true → 별 표시

&#x20;       public SessionAssets assets;

&#x20;   }



예시

{

&#x20;   "session\_id":          "session\_002",

&#x20;   "character\_npc\_name":  "뭉치",

&#x20;   "bubble\_text":         "안녕하세요!",

&#x20;   "likes":               12,

&#x20;   "is\_top\_liked":        true,

&#x20;   "assets": {

&#x20;       "character": { ... },

&#x20;       "object":    { ... }

&#x20;   }

}

💡`session\_id` — 이 관람객 세션의 고유 ID



\## WsEvent

&#x20;   public class WsEvent

&#x20;   {

&#x20;       public string @event;         // "session\_ready" | "likes\_updated"

&#x20;       public string session\_id;

&#x20;   }



💡역할

\- WebSocket을 통해 서버와 주고 받는 모든 이벤트는 이 구조를 공통으로 가진다.

{ "event": "session\_ready",  "session\_id": "session\_001" }

{ "event": "likes\_updated",  "session\_id": "session\_002" }



//event 예시

ev.@event == "session\_ready"   // 새 관람객 세션 준비됨

ev.@event == "likes\_updated"   // 누군가 좋아요 눌렀음



\## WsLikesEvent

WsEvent를 상속.

→ WsEvent의 필드를 그대로 물려받고 추가 필드만 선언함.

public class WsLikesEvent : WsEvent

{

&#x20;   public int    likes;

&#x20;   public string top\_session\_id;

}



💡서버가 “좋아요” 발생시 WebSocket으로 push하는 메시지.

{

&#x20;   "event":          "likes\_updated",

&#x20;   "session\_id":     "session\_002",

&#x20;   "likes":          13,

&#x20;   "top\_session\_id": "session\_002"

}



\## LikeResponse

LikeSystem.cs 에서 사용

public class LikeResponse

&#x20;   {

&#x20;       public string session\_id;

&#x20;       public int    likes;

&#x20;       public bool   is\_top\_liked;

&#x20;       public string top\_session\_id;

&#x20;   }



예시

{

&#x20;   "session\_id":     "session\_002",

&#x20;   "likes":          13,

&#x20;   "is\_top\_liked":   true,

&#x20;   "top\_session\_id": "session\_002"

}



LikeResponse VS WsLikesEvent

LikeResponse vs WsLikesEvent — 같은 정보, 다른 경로

좋아요 한 번 발생 시 두 가지가 동시에 옴:



관람객 A가 좋아요 클릭

&#x20;       │

&#x20;       ▼

&#x20;   POST /like

&#x20;       │

&#x20;  ┌────┴────┐

&#x20;  ▼         ▼

LikeResponse      WsLikesEvent

(HTTP 응답)        (WebSocket broadcast)

관람객 A만 수신    모든 클라이언트 수신

```



\## “좋아요” 눌렀을 때 일어나는 일

실제 흐름 — 좋아요 한 번 눌렸을 때



관람객이 좋아요 버튼 클릭

&#x20;       │

&#x20;       ▼

LikeSystem.OnLikePressed()

POST /sessions/session\_002/like

&#x20;       │

&#x20;       ▼

서버: DB likes +1, 1위 재계산

&#x20;       │

&#x20;       ├─→ LikeResponse 반환 (HTTP 응답)

&#x20;       │

&#x20;       └─→ WebSocket broadcast (모든 Unity 클라이언트에게)

&#x20;               │

&#x20;               ▼

&#x20;       WsLikesEvent 수신

&#x20;       {

&#x20;           event:          "likes\_updated",

&#x20;           session\_id:     "session\_002",   ← 좋아요 받은 세션

&#x20;           likes:          13,              ← 갱신된 좋아요 수

&#x20;           top\_session\_id: "session\_002"    ← 새 1위

&#x20;       }

&#x20;               │

&#x20;               ▼

&#x20;       PlazaManager.HandleLikesUpdated(ev)

&#x20;               │

&#x20;       ┌───────┴───────┐

&#x20;       ▼               ▼

&#x20; 해당 세션뷰         모든 세션뷰

&#x20; UpdateLikes(13)    SetTopLiked 재계산

&#x20; "♥ 13"            새 1위에만 ★



5\. 각 파일별로 그 안에 어떤 함수가 있는지, 어떤 필드들이 있는지 파악

\# 각 파일별 요약

\# Character/

\## MotionType.cs

💡

\- 모션 종류를 정의하는 enum

\- 모든 파일이 이 타입을 기준으로 모션을 취한다.

| 구분 | 이름 |

| --- | --- |

| enum 값 | `Idle, Dance, Run, Jump, Wave, Sit, Kick, Spin, Cheer, Clap` |



\## MixamoMotionLibrary.cs

💡

`MotionType → AnimationClip` 매핑 보관소(ScriptableObject, 에셋 파일로 저장)

| 구분 | 이름 | 설명 |

| --- | --- | --- |

| 내부 구조체 | `MotionEntry` | `motionType` + `clip` 쌍 |

| 필드 | `\_entries\[]` | Inspector에서 클립 연결하는 배열 |

| 캐시 | `\_cache` | `Dictionary<MotionType, AnimationClip>` 런타임 빠른 조회용 |



| 함수 | 역할 |

| --- | --- |

| `OnEnable()` | 앱 시작 시 캐시 자동 빌드 |

| `BuildCache()` | `\_entries` → `\_cache` 딕셔너리 변환 |

| `GetClip(MotionType)` | 타입에 맞는 클립 반환, 없으면 Idle 폴백 |

| `HasClip(MotionType)` | 해당 타입 클립 등록 여부 확인 |



\## MotionPromptParser.cs

💡

사용자 텍스트 입력 → MotionType 변환

| 구분 | 이름 | 설명 |

| --- | --- | --- |

| 필드 | `\_keywordMap` | `(키워드, MotionType)` 쌍 리스트 |



| 함수 | 역할 |

| --- | --- |

| `Parse(string input)` | 입력 문자열에서 키워드 검색 → MotionType 반환. 없으면 Idle |



\## CharacterAnimationController.cs

💡

캐릭터 Animator 제어 (walk/idle/Mixamo 모션 클립 런타임 교체)

| 구분 | 이름 | 설명 |

| --- | --- | --- |

| 필드 | `npcName` | NPC 이름 (Inspector 표시 + 로그) |

| 필드 | `bubbleText` | 말풍선 텍스트 |

| private | `\_animator` | Unity Animator 컴포넌트 참조 |

| private | `\_overrideController` | 클립 런타임 교체를 위한 AnimatorOverrideController |

| 상수 | `MOTION\_SLOT` | `"Motion"` — Override 대상 슬롯 이름 |



| 함수 | 역할 |

| --- | --- |

| `animation\_walk()` | walk Trigger 발동 |

| `animation\_idle()` | idle Trigger 발동 |

| `PlayAnimation(string)` | animationKey로 Trigger 발동 |

| `PlayMotionClip(AnimationClip)` | Mixamo 클립을 Motion 슬롯에 교체 후 재생 |

| `Initialize(CharacterAssetData, string)` | npcName, bubbleText 세팅 |

| `InitOverrideController()` | AnimatorOverrideController 초기화 |



\## PlacedCharacterController.cs

💡

광장 배치 캐릭터의 모션 재생 통합 컨트롤러 (프롬프트 → 모션까지 한 번에)

| 구분 | 이름 | 설명 |

| --- | --- | --- |

| 필드 | `motionLibrary` | MixamoMotionLibrary ScriptableObject 연결 |

| private | `\_animation` | CharacterAnimationController 참조 |



| 함수 | 역할 |

| --- | --- |

| `PlayMotionFromPrompt(string)` | 텍스트 입력 → Parse → GetClip → PlayMotionClip 전체 흐름 실행 |

| `PlayMotion(MotionType)` | MotionType 직접 지정해서 재생 (테스트용) |



\## GeneratedCharacterSpawner.cs

💡

SessionData로 캐릭터를 가이드/배치 두 역할로 스폰

| 구분 | 이름 | 설명 |

| --- | --- | --- |

| 필드 | `mockCharacterPrefab` | Mock용 FBX Prefab |

| 필드 | `guideSpawnPoint` | 가이드 NPC 시작 위치 |

| 필드 | `placedSpawnPoint` | 배치 캐릭터 위치 (오브제 옆) |

| private | `\_guideInstance` | 생성된 가이드 GameObject |

| private | `\_placedInstance` | 생성된 배치 캐릭터 GameObject |



| 함수 | 역할 |

| --- | --- |

| `SpawnGuide(SessionData)` | 가이드 NPC 생성 + Initialize → GuideNPCController 반환 |

| `SpawnPlaced(SessionData)` | 배치 캐릭터 생성 → GameObject 반환 |

| `SpawnCharacter(...)` | Mock/Server 분기 처리 내부 로직 |

| `SpawnFromServer(...)` | Server Mode FBX 런타임 로드 (TODO) |



\## GuideNPCController.cs

💡

가이드 시나리오 전체 진행(이동 + 말풍선 + 이벤트 발행)

| 구분 | 이름 | 설명 |

| --- | --- | --- |

| 필드 | `Animation` | CharacterAnimationController 참조 |

| 필드 | `moveSpeed, rotationSpeed, arrivalThreshold` | 이동 설정 |

| 필드 | `plazaPathWaypoints\[]` | 광장 이동 경로 |

| 필드 | `myCreationWaypoint` | 내 창작물 앞 위치 |

| 필드 | `placedCharacter` | PlacedCharacterController 참조 |

| private | `\_npcName, \_sessionId` | 초기화 후 내부 저장 |

| 이벤트 | `OnDialogueChanged` | 말풍선 텍스트 변경 시 발행 |

| 이벤트 | `OnGuideFinished` | 시나리오 완료 → 자유 모드 전환 |

| 이벤트 | `OnBubbleTextInputRequested` | 인사말 입력 요청 시 발행 |

| 이벤트 | `OnMotionPromptRequested` | 모션 입력 요청 시 발행 |



| 함수 | 역할 |

| --- | --- |

| `Initialize(SessionData)` | NPC 이름/세션ID 세팅 |

| `StartGuideScenario()` | 시나리오 코루틴 시작 |

| `Say(string)` | 말풍선 텍스트 출력 + 이벤트 발행 |

| `MoveTo(Vector3)` | 목표 위치로 이동 시작 |

| `WaitUntilArrived()` | 이동 완료 대기 (yield return용) |

| `StopMoving()` | 이동 즉시 중단 |

| `GuideScenarioRoutine()` | 1\~8단계 시나리오 본문 코루틴 |

| `MoveRoutine(Vector3)` | 실제 이동 처리 코루틴 |



\# Core/

\## ServerConfig.cs

💡

서버 주소 설정 및 API URL 생성 (전시 환경에서 IP 교체)

| 구분 | 이름 | 설명 |

| --- | --- | --- |

| 프로퍼티 | `BaseUrl` | `http://IP:8000` (JSON 또는 기본값) |

| 프로퍼티 | `WsUrl` | `ws://IP:8000/ws/unity` |

| 함수 | `UnitySessionUrl(sid)` | `/unity/sessions/{sid}` |

| 함수 | `PlazaSessionsUrl()` | `/unity/plaza/sessions` |

| 함수 | `LikeUrl(sid)` | `/sessions/{sid}/like` |

| 함수 | `BubbleTextUrl(sid)` | `/sessions/{sid}/profile` |

| 함수 | `LoadBaseUrl()` | `StreamingAssets/Config/server\_config.json` 읽기 |



\# Data/

\## SessionData.cs

💡클래스



\# Manager/

\## DataSourceMode.cs

💡Mock/Server 모드 분기용 enum

| enum 값 | 설명 |

| --- | --- |

| `Mock` | JSON 파일 사용 (개발용) |

| `Server` | FastAPI 서버 연동 (전시용) |



\## SessionManager.cs

💡

세션 데이터 로드 진입점, 전체에 OnSessionLoaded 이벤트 브로드캐스트

→ 서버 → Json → SessionData 클래스 필드(SessionData 클래스)를 받아서 유니티에 로드 해주는 역할

| 구분 | 이름 | 설명 |

| --- | --- | --- |

| 싱글턴 | `Instance` | 씬 전환 후에도 유지 |

| 필드 | `dataSourceMode` | Mock / Server 선택 |

| 필드 | `sessionId` | Server Mode 세션 ID |

| 필드 | `\_apiClient` | ApiClient 참조 |

| 프로퍼티 | `CurrentSession` | 현재 로드된 SessionData |

| 이벤트 | `OnSessionLoaded` | 로드 완료 시 전체 브로드캐스트 |



| 함수 | 역할 |

| --- | --- |

| `Start()` | 앱 시작 시 자동 로드 |

| `LoadSession(sid, onLoaded)` | Mock/Server 분기 후 데이터 로드 |

| `Apply(data, onLoaded)` | CurrentSession 저장 + 이벤트 발행 |



\# Mock/

\## MockSessionLoader.cs

💡

`Resources/Mock/mock\_session.json` 읽어서 SessionData 반환

→Mocking 캐릭터용 Json

| 구분 | 이름 | 설명 |

| --- | --- | --- |

| 상수 | `ResourcePath` | `"Mock/mock\_session"` |



| 함수 | 역할 |

| --- | --- |

| `Load()` | JSON 읽기 → JsonUtility.FromJson → SessionData 반환 |



\# Network/

\## ApiClient.cs

💡

FastAPI 서버 REST API 호출(GET,POST,PATCH)

| 구분 | 이름 | 설명 |

| --- | --- | --- |

| 싱글턴 | `Instance` |  |

| 내부 클래스 | `SessionStatusResponse` | `session\_id`, `ready\_for\_unity` |



| 함수 | 역할 |

| --- | --- |

| `FetchUnitySession(sid, onSuccess)` | GET 현재 세션 데이터 |

| `PollUntilReady(sid, onReady)` | `ready\_for\_unity == true` 될 때까지 3초마다 polling |

| `FetchPlazaSessions(onSuccess)` | GET 광장 전체 세션 목록 |

| `LikeSession(sid, onSuccess)` | POST 좋아요 +1 |

| `UpdateBubbleText(sid, text, onSuccess)` | PATCH 인사말 서버 저장 |



\## WebSocketManager.cs

💡

&#x20;서버 WebSocket 연결 관리 + 메시지 파싱 + 이벤트 발행 + 자동 재연결

| 구분 | 이름 | 설명 |

| --- | --- | --- |

| 싱글턴 | `Instance` |  |

| 이벤트 | `OnSessionReady` | 새 세션 준비 완료 시 |

| 이벤트 | `OnLikesUpdated` | 좋아요 갱신 시 |

| 필드 | `reconnectDelay` | 재연결 대기 시간 (기본 3초) |

| private | `\_ws` | NativeWebSocket 인스턴스 |

| private | `\_running, \_reconnecting` | 연결 상태 플래그 |



| 함수 | 역할 |

| --- | --- |

| `StartListening()` | WS 연결 시작 |

| `StopListening()` | WS 연결 종료 |

| `ConnectWs()` | 실제 연결 + 이벤트 핸들러 등록 |

| `HandleMessage(string)` | JSON 파싱 → 이벤트 타입별 분기 발행 |

| `ScheduleReconnect()` | 끊김 감지 후 재연결 예약 |

| `OnOpen/OnMessage/OnError/OnClose` | WS 콜백 핸들러 |

| `Update()` | 매 프레임 메시지 큐 dispatch |



\# Object/

\## GenerateObjectSpawner.cs

💡

세션 데이터로 오브제 스폰 (Mock: Prefab / Server: GLB 다운로드)

| 구분 | 이름 | 설명 |

| --- | --- | --- |

| 필드 | `objectPrefab` | Mock용 기본 Prefab |

| 필드 | `spawnPoint` | 배치 위치 |

| private | `\_spawnedObject` | 현재 스폰된 오브제 |



| 함수 | 역할 |

| --- | --- |

| `Spawn(ObjectAssetData)` | model\_url 유무로 Mock/Server 분기 후 오브제 생성 |



\# Plaza/

\## PlazaManager.cs

💡

자유 모드 광장 전체 관리 (스폰 + 실시간 좋아요 갱신)

| 구분 | 이름 | 설명 |

| --- | --- | --- |

| 싱글턴 | `Instance` |  |

| 필드 | `spawnPoints\[]` | 광장 배치 위치 배열 |

| 필드 | `mockCharacterPrefab, mockObjectPrefab` | Mock Prefab |

| 필드 | `sessionViewPrefab` | PlazaSessionView 프리팹 |

| private | `\_views` | 생성된 PlazaSessionView 목록 |

| private | `\_topSessionId` | 현재 1위 세션 ID |



| 함수 | 역할 |

| --- | --- |

| `EnterPlaza()` | 광장 진입 — 데이터 로드 + WS 구독 |

| `ExitPlaza()` | 광장 退장 — WS 구독 해제 + 오브젝트 제거 |

| `LoadAndSpawnPlaza()` | Mock/Server 분기 후 세션뷰 전체 생성 코루틴 |

| `SpawnSessionAssets(session, point)` | 캐릭터+오브제 배치 |

| `HandleLikesUpdated(WsLikesEvent)` | WS 수신 시 해당 세션뷰 좋아요/별 즉시 갱신 |

| `LoadMockPlaza()` | mock\_plaza.json 로드 |



\## PlazaSessionView.cs

💡

광장 내 세션 하나의 UI 표현 (좋아요 수 / 말풍선 / 별 표시)

| 구분 | 이름 | 설명 |

| --- | --- | --- |

| 프로퍼티 | `SessionId` | 이 뷰가 담당하는 세션 ID |

| 필드 | `likesCountText` | TextMeshPro — ♥ N |

| 필드 | `bubbleText` | TextMeshPro — 인사말 |

| 필드 | `starObject` | 1위 별 표시 GameObject |

| private | `\_likeSystem` | LikeSystem 컴포넌트 참조 |



| 함수 | 역할 |

| --- | --- |

| `Initialize(PlazaSessionData, bool isTop)` | 세션 데이터 주입 + 초기 UI 세팅 |

| `UpdateLikes(int)` | 좋아요 수 텍스트 갱신 |

| `SetTopLiked(bool)` | 별 표시 켜기/끄기 |



\## LikeSystem.cs

💡

관람객 근접 감지 + 좋아요 POST + UI 피드백

| 구분 | 이름 | 설명 |

| --- | --- | --- |

| 필드 | `triggerRadius` | 좋아요 가능 범위 (기본 2m) |

| 필드 | `likeButtonUI` | 근접 시 표시할 하트/버튼 UI |

| private | `\_sessionId, \_likes` | 담당 세션 정보 |

| private | `\_playerNearby` | 플레이어 근접 여부 플래그 |



| 함수 | 역할 |

| --- | --- |

| `Initialize(sessionId, likes)` | 세션 ID + 초기 좋아요 수 세팅 |

| `UpdateCount(int)` | 좋아요 수 내부 갱신 |

| `OnLikePressed()` | 버튼 클릭 시 호출 → SendLike 코루틴 시작 |

| `SendLike()` | POST /like → LikeResponse로 UI 즉시 갱신 |

| `OnTriggerEnter/Exit` | 플레이어 근접/이탈 감지 → 버튼 UI 표시/숨김 |

