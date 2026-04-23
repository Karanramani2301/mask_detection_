def calculate_compliance(prediction, class_labels=['Mask Proper', 'Mask Improper', 'No Mask']):
    """
    Computes a complex Compliance Score (0-100) and Final Safety Status.
    
    Args:
        prediction (list/ndarray): Array of confidence probabilities [prop, improp, no].
        class_labels: Label mapping array.
        
    Returns:
        dict: Containing the best label, max confidence, calculated score, and final status.
    """
    import numpy as np
    
    max_idx = np.argmax(prediction)
    base_confidence = prediction[max_idx]
    best_label = class_labels[max_idx]
    
    score = 0.0
    
    # Base scoring multipliers
    if best_label == 'Mask Proper':
        score = base_confidence * 100
    elif best_label == 'Mask Improper':
        score = base_confidence * 60
    else: # No Mask
        score = base_confidence * 20
        
    # Penalty for low confidence
    if base_confidence < 0.6:
        # Heavily penalize uncertain captures
        score -= 20
        
    # Bound limits
    score = max(0, min(100, score))
    
    # Determine the Final Final Status text Status
    if score >= 80:
        status = "SAFE"
    elif score < 50:
        status = "VIOLATION"
    else:
        # Score between 50 and 80, OR confidence was just very low
        status = "UNCERTAIN"
        
    if base_confidence < 0.6:
        status = "UNCERTAIN"

    return {
        "label": best_label,
        "confidence": round(base_confidence * 100, 2),
        "score": round(score, 2),
        "status": status
    }
