def ml_policy(row):

    rows = pd.DataFrame(
        [row[feature_cols[:-1]].to_dict() for _ in actions]
    )

    rows["recovery_action"] = actions

    probabilities = model.predict_proba(
        rows[feature_cols]
    )[:, 1]

    net_ev = (
        probabilities * row["amount"]
        - np.array([action_cost[a] for a in actions])
    )

    best_idx = net_ev.argmax()

    return {
        "suggested_action": actions[best_idx],
        "predicted_probability": float(probabilities[best_idx]),
        "expected_net_recovery": float(net_ev[best_idx]),
        "all_action_scores": dict(
            zip(actions, probabilities)
        )
    }