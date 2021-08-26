def predict_prob(model,texts):
    probs = []
    for t in texts:
        probs.append(model.predict([t]))
    return probs

