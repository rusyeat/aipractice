tree = {
    "A": ["B", "C"], "B": ["D", "E"], "C": ["F", "G"],
    "D": ["H", "I"], "E": ["J", "K"],
    "F": ["L", "M"], "G": ["N", "O"]
}
payoffs = {"H": 3, "I": 12, "J": 8, "K": 2,
           "L": 4, "M": 6, "N": 14, "O": 5}


def is_terminal(state):
    return state in payoffs


def successors(state, reverse=False):
    children = tree[state]
    return children[::-1] if reverse else children[:]


def utility(state):
    return payoffs[state]


def minimax(state, maximizing, counter, reverse=False):
    counter["calls"] += 1
    if is_terminal(state):
        return utility(state)
    values = []
    for child in successors(state, reverse):
        values.append(minimax(child, not maximizing, counter, reverse))
    return max(values) if maximizing else min(values)


def alphabeta(state, alpha, beta, maximizing, counter, reverse=False):
    counter["calls"] += 1
    counter["visited"].append(state)
    if is_terminal(state):
        return utility(state)
    row = {"node": state, "alpha": alpha, "beta": beta,
           "children": [], "pruned": []}
    counter["trace"].append(row)
    value = float("-inf") if maximizing else float("inf")
    for i, child in enumerate(successors(state, reverse)):
        row["children"].append(child)
        result = alphabeta(child, alpha, beta, not maximizing,
                           counter, reverse)
        if maximizing:
            value = max(value, result)
            alpha = max(alpha, value)
        else:
            value = min(value, result)
            beta = min(beta, value)
        if alpha >= beta:
            row["pruned"] = successors(state, reverse)[i + 1:]
            break
    row["value"] = value
    return value


def experiment(reverse=False, verbose=True):
    mm = {"calls": 0}
    ab = {"calls": 0, "visited": [], "trace": []}
    m = minimax("A", True, mm, reverse)
    a = alphabeta("A", float("-inf"), float("inf"), True, ab, reverse)
    all_nodes = list(tree) + list(payoffs)
    skipped = [s for s in all_nodes if s not in ab["visited"]]
    # Түбір балаларының есептелген нәтижесінен жүрісті таңдау.
    child_values = {r["node"]: r["value"] for r in ab["trace"]
                    if r["node"] in tree["A"]}
    move = max(child_values, key=child_values.get)
    assert m == a == 8 and move == "B"
    assert mm["calls"] == 15
    assert ab["calls"] + len(skipped) == 15
    if verbose:
        print("Order:", "right-to-left" if reverse else "left-to-right")
        print("Depth: 3")
        print("Minimax:", m, "calls:", mm["calls"])
        print("Alpha-Beta:", a, "calls:", ab["calls"])
        print("Move:", move)
        print("Visited:", ", ".join(ab["visited"]))
        print("Pruned nodes:", ", ".join(skipped), "count:", len(skipped))
        print()
    return ab


# D бөліміне арналған шартты күйлер, негізгі ағаштың жапырақтары емес.
# Белгілер реті: material, position, mobility.
features = {"X": (6, 4, 1), "Y": (3, 3, 7)}


def evaluate(state, weights):
    material, position, mobility = features[state]
    return (weights[0] * material + weights[1] * position
            + weights[2] * mobility)


def evaluation_experiment():
    for weights in [(0.5, 0.3, 0.2), (0.3125, 0.1875, 0.5)]:
        assert abs(sum(weights) - 1) < 1e-9
        scores = {s: evaluate(s, weights) for s in features}
        print("Weights:", weights)
        print("X:", round(scores["X"], 4),
              "Y:", round(scores["Y"], 4))
        print("Preferred:", max(scores, key=scores.get))


def main():
    experiment(False)
    experiment(True)
    evaluation_experiment()


if __name__ == "__main__":
    main()
