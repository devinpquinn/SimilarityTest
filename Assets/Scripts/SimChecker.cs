using UnityEngine;

public class SimChecker : MonoBehaviour
{
    public string sentenceA;
    public string sentenceB;
    
    public void CheckSimilarity()
    {
        float similarity = SimilarityHandler.GetSimilarity(sentenceA, sentenceB);
        string hex = ColorUtility.ToHtmlStringRGB(Color.Lerp(Color.red, Color.green, Mathf.Clamp01(similarity)));
        Debug.Log($"Similarity between \"{sentenceA}\" and \"{sentenceB}\": <color=#{hex}>{similarity}</color>");
    }
}
