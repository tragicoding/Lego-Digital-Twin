using UnityEngine;
using LegoTwin.World;

namespace LegoTwin.Core
{
    /// <summary>
    /// 플레이어 리그(XR Origin / Camera Rig 루트)를 안전하게 순간이동시키는 공용 유틸.
    ///
    /// CharacterController가 활성 상태에서 transform.position을 직접 바꾸면 내부 위치가
    /// 어긋나 이후 Move()가 제자리에 묶인다. → CC를 끄고 → 위치 설정 → 다시 켜서
    /// 내부 위치까지 동기화한다.
    ///
    /// 텔레포트가 필요한 모든 곳(가이드 시나리오, 지도 UI 등)이 이 한 메서드를 재사용한다.
    /// (중복 구현 금지 — CC-safe 로직의 단일 소스)
    /// </summary>
    public static class PlayerTeleporter
    {
        /// <summary>
        /// rig 를 position 으로 이동한다. rotation 을 주면 회전도 적용한다.
        /// rig 가 null 이면 아무 것도 하지 않는다.
        /// </summary>
        public static void Teleport(Transform rig, Vector3 position, Quaternion? rotation = null)
        {
            if (rig == null) return;

            var cc = rig.GetComponent<CharacterController>();
            if (cc == null) cc = rig.GetComponentInChildren<CharacterController>();

            bool reenable = cc != null && cc.enabled;
            if (reenable) cc.enabled = false;

            rig.position = position;
            if (rotation.HasValue) rig.rotation = rotation.Value;

            if (reenable) cc.enabled = true;

            // CC 비활성 이동은 OnTriggerExit 를 발생시키지 않으므로 모든 WorldButton 힌트를 강제 숨김
            WorldButton.ForceHideAll();
        }
    }
}
