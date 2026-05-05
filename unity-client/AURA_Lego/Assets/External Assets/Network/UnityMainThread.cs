using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 백그라운드 스레드에서 Unity 메인 스레드로 작업을 넘기는 헬퍼.
/// BackendClient의 async 수신 루프에서 사용한다.
/// </summary>
public class UnityMainThread : MonoBehaviour
{
    private static readonly Queue<Action> _queue = new Queue<Action>();
    private static readonly object _lock = new object();

    private void Update()
    {
        while (true)
        {
            Action action;
            lock (_lock)
            {
                if (_queue.Count == 0) break;
                action = _queue.Dequeue();
            }
            action?.Invoke();
        }
    }

    public static void Enqueue(Action action)
    {
        lock (_lock) { _queue.Enqueue(action); }
    }
}
