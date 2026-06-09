using UnityEngine;

namespace LegoTwin.UI
{
    /// <summary>GameObject를 지정한 축으로 천천히 회전시킨다.</summary>
    public class RotateObject : MonoBehaviour
    {
        [Tooltip("초당 회전 각도")]
        [SerializeField] private float speed = 30f;

        [Tooltip("회전 축 (기본: Y축)")]
        [SerializeField] private Vector3 axis = Vector3.up;

        private void Update()
        {
            transform.Rotate(axis * (speed * Time.deltaTime));
        }
    }
}
