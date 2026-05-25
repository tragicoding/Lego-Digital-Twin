using System;
using System.Collections;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace LegoTwin.Motion
{
    /// <summary>
    /// 실시간 모션 캡처 수신기
    ///
    /// [Mode A] UDP 실시간  : realtime_motion.py 가 웹캠으로 캡처한 동작을 즉시 반영
    /// [Mode B] 파일 재생   : bp_to_npc.py 로 생성한 motion.txt 를 루프 재생
    ///
    /// UDP 패킷 포맷 (1줄 텍스트):
    ///   root_x root_y root_z  qx0 qy0 qz0 qw0  qx1 ... qw23
    ///   = 3 + 4*24 = 99 floats, 공백 구분, \n 종료
    ///
    /// Bone 순서 (npc_skeleton.txt):
    ///   0=Hip, 1=Pelvis,
    ///   2=L_Thigh, 3=L_Calf, 4=L_Foot, 5=L_ToeBase,
    ///   6=R_Thigh, 7=R_Calf, 8=R_Foot, 9=R_ToeBase,
    ///   10=Waist, 11=Spine01, 12=Spine02,
    ///   13=NeckTwist01, 14=NeckTwist02, 15=Head,
    ///   16=L_Clavicle, 17=L_Upperarm, 18=L_Forearm, 19=L_Hand,
    ///   20=R_Clavicle, 21=R_Upperarm, 22=R_Forearm, 23=R_Hand
    /// </summary>
    public class BlazeMotionReceiver : MonoBehaviour
    {
        // ── 모드 선택 ────────────────────────────────────────────────────────
        public enum Mode { UDP_Realtime, FilePlayback }

        [Header("Mode")]
        public Mode mode = Mode.UDP_Realtime;

        // ── UDP 설정 ─────────────────────────────────────────────────────────
        [Header("UDP (Mode A)")]
        [Tooltip("수신 포트 — realtime_motion.py 의 --port 와 일치시킬 것")]
        public int udpPort = 7777;

        // ── 파일 재생 설정 ────────────────────────────────────────────────────
        [Header("File Playback (Mode B)")]
        public string motionFilePath = "rigmotion/toobad_npc_motion.txt";
        public float  fileFrameRate  = 30f;
        public bool   loop           = true;

        // ── 캐릭터 설정 ──────────────────────────────────────────────────────
        [Header("Character")]
        [Tooltip("캐릭터 루트 오브젝트 (Hip 의 부모 Transform)")]
        public Transform characterRoot;

        [Tooltip("아래 배열을 수동으로 채우거나 'Auto-Assign Bones' 를 사용하세요")]
        public Transform[] bones = new Transform[NumJoints];

        // ── 조정 ─────────────────────────────────────────────────────────────
        [Header("Tuning")]
        [Tooltip("포지션 스케일 (캐릭터 크기에 맞게 조정)")]
        public float positionScale  = 1.0f;
        [Tooltip("루트 위치 오프셋 (월드 공간)")]
        public Vector3 rootOffset   = Vector3.zero;
        [Tooltip("쿼터니언 스무딩 (0=꺼짐, 1=최대)")]
        [Range(0f, 0.95f)]
        public float smoothing      = 0.3f;

        // ────────────────────────────────────────────────────────────────────
        private const int NumJoints = 24;

        private static readonly string[] BoneNames = {
            "Hip", "Pelvis",
            "L_Thigh", "L_Calf", "L_Foot", "L_ToeBase",
            "R_Thigh", "R_Calf", "R_Foot", "R_ToeBase",
            "Waist", "Spine01", "Spine02",
            "NeckTwist01", "NeckTwist02", "Head",
            "L_Clavicle", "L_Upperarm", "L_Forearm", "L_Hand",
            "R_Clavicle", "R_Upperarm", "R_Forearm", "R_Hand"
        };

        // ── 내부 상태 ─────────────────────────────────────────────────────────
        // UDP
        private UdpClient          _udpClient;
        private Thread             _udpThread;
        private ConcurrentQueue<string> _packetQueue = new ConcurrentQueue<string>();
        private bool               _running;

        // 파일 재생
        private string[]  _fileLines;
        private int       _fileFrame;
        private Coroutine _fileCoroutine;

        // 스무딩
        private Quaternion[] _prevRots;

        // ============================================================
        void Awake()
        {
            if (bones == null || bones.Length != NumJoints)
                bones = new Transform[NumJoints];
            _prevRots = new Quaternion[NumJoints];
            for (int i = 0; i < NumJoints; i++)
                _prevRots[i] = Quaternion.identity;
        }

        void Start()
        {
            if (mode == Mode.UDP_Realtime)
                StartUDP();
            else
                StartFilePlayback();
        }

        void Update()
        {
            // UDP 큐에서 최신 패킷만 처리 (오래된 것은 버림)
            if (mode == Mode.UDP_Realtime)
            {
                string latest = null;
                while (_packetQueue.TryDequeue(out string pkt))
                    latest = pkt;
                if (latest != null)
                    ApplyFrame(latest);
            }
        }

        void OnDestroy()
        {
            StopUDP();
            if (_fileCoroutine != null) StopCoroutine(_fileCoroutine);
        }

        // ============================================================
        // UDP 모드
        // ============================================================
        private void StartUDP()
        {
            try
            {
                _udpClient = new UdpClient(udpPort);
                _running   = true;
                _udpThread = new Thread(UDPReceiveLoop) { IsBackground = true };
                _udpThread.Start();
                Debug.Log($"[BlazeMotionReceiver] UDP 수신 시작 — port {udpPort}");
            }
            catch (Exception e)
            {
                Debug.LogError($"[BlazeMotionReceiver] UDP 시작 실패: {e.Message}");
            }
        }

        private void StopUDP()
        {
            _running = false;
            _udpClient?.Close();
            _udpThread?.Join(500);
        }

        private void UDPReceiveLoop()
        {
            IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);
            while (_running)
            {
                try
                {
                    byte[] data   = _udpClient.Receive(ref remote);
                    string packet = Encoding.UTF8.GetString(data).Trim();
                    if (!string.IsNullOrEmpty(packet))
                        _packetQueue.Enqueue(packet);
                }
                catch (SocketException)
                {
                    // 소켓 닫힘 시 정상 종료
                    break;
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[BlazeMotionReceiver] UDP 수신 오류: {e.Message}");
                }
            }
        }

        // ============================================================
        // 파일 재생 모드
        // ============================================================
        private void StartFilePlayback()
        {
            string path = motionFilePath;
            if (!Path.IsPathRooted(path))
                path = Path.Combine(Application.streamingAssetsPath, motionFilePath);
            if (!File.Exists(path))
                path = Path.Combine(Application.dataPath, "..", motionFilePath);

            if (!File.Exists(path))
            {
                Debug.LogError($"[BlazeMotionReceiver] 파일 없음: {motionFilePath}");
                return;
            }

            _fileLines  = File.ReadAllLines(path);
            _fileFrame  = 0;
            _fileCoroutine = StartCoroutine(FilePlaybackLoop());
            Debug.Log($"[BlazeMotionReceiver] 파일 재생: {_fileLines.Length} frames");
        }

        private IEnumerator FilePlaybackLoop()
        {
            float interval = 1f / fileFrameRate;
            while (true)
            {
                if (_fileFrame >= _fileLines.Length)
                {
                    if (loop) _fileFrame = 0;
                    else yield break;
                }
                ApplyFrame(_fileLines[_fileFrame++]);
                yield return new WaitForSeconds(interval);
            }
        }

        // ============================================================
        // 프레임 적용
        // ============================================================
        private void ApplyFrame(string line)
        {
            if (string.IsNullOrWhiteSpace(line)) return;

            var tokens = line.Trim().Split(new[]{' ', '\t'},
                                           StringSplitOptions.RemoveEmptyEntries);
            int expected = 3 + 4 * NumJoints;
            if (tokens.Length < expected)
            {
                Debug.LogWarning($"[BlazeMotionReceiver] 토큰 수 부족 ({tokens.Length}/{expected})");
                return;
            }

            // 루트 위치
            var rootPos = new Vector3(
                F(tokens[0]) * positionScale,
                F(tokens[1]) * positionScale,
                F(tokens[2]) * positionScale
            ) + rootOffset;

            if (characterRoot != null)
                characterRoot.localPosition = rootPos;

            // 관절 쿼터니언
            for (int j = 0; j < NumJoints; j++)
            {
                int o  = 3 + j * 4;
                var q  = new Quaternion(F(tokens[o]), F(tokens[o+1]),
                                        F(tokens[o+2]), F(tokens[o+3]));

                if (smoothing > 0f)
                    q = Quaternion.Slerp(_prevRots[j], q, 1f - smoothing);
                _prevRots[j] = q;

                if (bones[j] != null)
                    bones[j].rotation = q;
            }
        }

        private static float F(string s) =>
            float.TryParse(s,
                System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out float v) ? v : 0f;

        // ============================================================
        // Inspector 유틸리티
        // ============================================================
        [ContextMenu("Auto-Assign Bones")]
        public void AutoAssignBones()
        {
            if (characterRoot == null)
            {
                Debug.LogWarning("[BlazeMotionReceiver] characterRoot 를 먼저 설정하세요.");
                return;
            }
            bones = new Transform[NumJoints];
            for (int i = 0; i < NumJoints; i++)
            {
                var t = FindDeep(characterRoot, BoneNames[i]);
                bones[i] = t;
                if (t == null)
                    Debug.LogWarning($"  본 없음: {BoneNames[i]}");
            }
            Debug.Log("[BlazeMotionReceiver] Auto-Assign 완료");
        }

        private static Transform FindDeep(Transform root, string name)
        {
            if (root.name == name) return root;
            foreach (Transform child in root)
            {
                var found = FindDeep(child, name);
                if (found != null) return found;
            }
            return null;
        }

#if UNITY_EDITOR
        void OnDrawGizmosSelected()
        {
            if (bones == null) return;
            Gizmos.color = Color.yellow;
            foreach (var b in bones)
                if (b != null) Gizmos.DrawWireSphere(b.position, 0.015f);
        }
#endif
    }
}
