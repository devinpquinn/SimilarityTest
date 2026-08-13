using System.Diagnostics;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class SimChecker : MonoBehaviour
{
    public string sentenceA;
    public string sentenceB;
    
    public void CheckSimilarity()
    {
        var stopwatch = Stopwatch.StartNew();
        float similarity = SimilarityHandler.GetSimilarity(sentenceA, sentenceB);
        stopwatch.Stop();

        string hex = ColorUtility.ToHtmlStringRGB(Color.Lerp(Color.red, Color.green, Mathf.Clamp01(similarity)));
        
        Debug.Log($"Similarity between \"<i>{sentenceA}</i>\" and \"<i>{sentenceB}</i>\": <color=#{hex}>{similarity}</color> (took {stopwatch.Elapsed.TotalMilliseconds:F2} ms)");
    }
}
