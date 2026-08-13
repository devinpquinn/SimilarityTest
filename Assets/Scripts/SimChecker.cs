using UnityEngine;

public class SimChecker : MonoBehaviour
{
    public string sentenceA;
    public string sentenceB;
    
    public void CheckSimilarity()
    {
        float similarity = SimilarityHandler.GetSimilarity(sentenceA, sentenceB);
        Debug.Log($"Similarity between \"{sentenceA}\" and \"{sentenceB}\": {similarity}");
    }
}
